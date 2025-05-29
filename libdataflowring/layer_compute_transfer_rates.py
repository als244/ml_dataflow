def get_total_flops(M, D, K, E_S, E_R, A_E_R, F_E, S, L_max, repeats):
	
	q = 2 * M * D * D
	kv = 2 * 2 * M * D * K

	## this chunk is broken into multiple seqs
	if (M + S) > L_max:
		a = 0
		leftover_M = L_max - S
		a += 2 * leftover_M * L_max * D
		remain_M = M - leftover_M
		while remain_M > 0:
			if (remain_M > L_max):
				a += 2 * L_max * L_max * D
				remain_M -= L_max
			else:
				a += 2 * remain_M * remain_M * D
				remain_M = 0
	## continuing attention on next tokens from past sequence
	else:
		total_seq_len = M + S
		new_queries = M
		a = 2 * new_queries * D * total_seq_len


	a_o = 2 * M * D * D


	shared_experts = E_S * 3 * 2 * M * D * F_E

	routed_experts = A_E_R * 3 * 2 * M * D * F_E

	ffn = shared_experts + routed_experts
	layer_flops = q + kv + a + a_o + ffn
	return layer_flops * repeats


def get_avg_chunk_per_routed_expert(M, E_R, A_E_R):
	if E_R == 0:
		return 0
	return (M * A_E_R) / E_R

def get_routed_expert_intensity(M, D, F_E, E_R, A_E_R, dtype_bytes):
	avg_m_per_expert = get_avg_chunk_per_routed_expert(M, E_R, A_E_R)

	routed_exp_flops = 2 * avg_m_per_expert * D * F_E
	routed_exp_bytes = dtype_bytes * (avg_m_per_expert * D + D * F_E + avg_m_per_expert * F_E)
	return routed_exp_flops / routed_exp_bytes


def get_layer_bytes_read(M, D, K, E_S, E_R, A_E_R, F_E, S, L_max, repeats, dtype_bytes):

	q_proj_bytes = dtype_bytes * (M * D + D * D + M * D)
	kv_proj_bytes = dtype_bytes * (2 * (M * D + D * K + M * K))

	## all the new queries
	attn_bytes = dtype_bytes * M * D

	## assumes S > L_max so we will read in all past k,v
	attn_bytes += dtype_bytes * 2 * S * K

	## now read in the remaining KV from this chunk
	attn_bytes += dtype_bytes * 2 * M * K

	a_o_proj_bytes = dtype_bytes * (M * D + D * D + M * D)


	shared_experts_bytes = dtype_bytes * (E_S * (3 * (M * D + D * F_E + M * F_E)))

	avg_m_per_expert = get_avg_chunk_per_routed_expert(M, E_R, A_E_R)

	routed_experts_bytes = dtype_bytes * (E_R * (3 * (avg_m_per_expert * D + D * F_E + avg_m_per_expert * F_E)))

	ffn_bytes = shared_experts_bytes + routed_experts_bytes

	total_bytes = q_proj_bytes + kv_proj_bytes + attn_bytes + a_o_proj_bytes + ffn_bytes

	return total_bytes * repeats


def get_unique_bytes_read(layer_bytes_read, layer_size, repeats):

	return layer_size + (layer_bytes_read - (layer_size * repeats))


def get_overall_layer_intensity(layer_flops, layer_bytes_read):
	return layer_flops / layer_bytes_read

def get_temporal_intensity(total_layer_flops, unique_bytes_read):
	return total_layer_flops / unique_bytes_read


def get_total_compute_time(layer_flops, compute_rate_tflops_per_sec):
	return layer_flops / compute_rate_tflops_per_sec


def get_layer_size(dtype_bytes, D, K, E_S, E_R, F_E):

	q = dtype_bytes * D * D
	kv = 2 * dtype_bytes * D * K
	a_o = dtype_bytes * D * D
	ffn = 3 * dtype_bytes * (E_S + E_R) * D * F_E
	return q + kv + a_o + ffn


def get_layer_transfer_time(layer_size_bytes, transfer_rate_bytes_per_sec):

	return layer_size_bytes / transfer_rate_bytes_per_sec



llama_1b = {"D": 2048, "K": 512, "E_S": 1, "E_R": 0, "A_E_R": 0, "F_E": 8192}
llama_8b = {"D": 4096, "K": 1024, "E_S": 1, "E_R": 0, "A_E_R": 0, "F_E": 14336}
llama_70b = {"D": 8192, "K": 1024, "E_S": 1, "E_R": 0, "A_E_R": 0, "F_E": 28672}
qwen_32b = {"D": 5120, "K": 640, "E_S": 1, "E_R": 0, "A_E_R": 0, "F_E": 27648}
deepseek_v3 = {"D": 7168, "K": 512, "E_S": 1, "E_R": 256, "A_E_R": 8, "F_E": 2048}


model = llama_8b



## model dimension
D = model["D"]
## key/value dimension
K = model["K"]

## number of shared experts
E_S = model["E_S"]
## total number of routed experts
E_R = model["E_R"]

## total number of active experts
A_E_R = model["A_E_R"]

## expert feed forward dimension
F_E = model["F_E"]



## dealing with seq lengths

## ASSUMES L_max > S

## prior seq len (if long seq and we have prior context saved)
S = 0

## max attended tokens (if M + S) > L_max, then remainder of M is chunked up
## into <= L_max sized independent windows
L_max = 2048



#### Pipeline configuration


## setting chunk size
M = 2048

## if mutliple chunks pass through the same layer
repeats = 8


total_tokens_per_layer = M * repeats

#### CALCULATIONS!



layer_flops = get_total_flops(M, D, K, E_S, E_R, A_E_R, F_E, S, L_max, repeats)

dtype_bytes = 2


layer_bytes_read = get_layer_bytes_read(M, D, K, E_S, E_R, A_E_R, F_E, S, L_max, repeats, dtype_bytes)


layer_size_bytes = get_layer_size(dtype_bytes, D, K, E_S, E_R, F_E) 

unique_bytes_read = get_unique_bytes_read(layer_bytes_read, layer_size_bytes, repeats)

avg_m_per_expert = get_avg_chunk_per_routed_expert(M, E_R, A_E_R)

routed_exp_intensity = get_routed_expert_intensity(M, D, F_E, E_R, A_E_R, dtype_bytes)


overall_intensity = get_overall_layer_intensity(layer_flops, layer_bytes_read)

temporal_intensity = get_temporal_intensity(layer_flops, unique_bytes_read)


compute_rate_tflops_per_sec = 650 * 1e12

layer_compute_time = get_total_compute_time(layer_flops, compute_rate_tflops_per_sec)




transfer_rate_bytes_per_sec = 40 * 1e9

layer_transfer_time = get_layer_transfer_time(layer_size_bytes, transfer_rate_bytes_per_sec)


print(f"\nModel Inputs:\n\tModel Dim = {D}\n\tKV Dim = {K}\n\tExpert Dim = {F_E}\n\t# Shared Experts: {E_S}\n\t# Routed Experts: {E_R}\n\t# Active-Routed Experts: {A_E_R}\n\tDtype Bytes = {dtype_bytes}\n\nData Inputs:\n\tPrior Seq Len = {S}\n\tChunk Size = {M}\n\tMax Attended Tokens = {L_max}\n\tRepeats (different chunks, same layer) = {repeats}\n\tTotal Tokens Processed by Layer: {total_tokens_per_layer}\n\nHardware Inputs:\n\tCompute Rate (TFLOPs) = {round(compute_rate_tflops_per_sec / 1e12, 0)}\n\tTransfer Rate (GB/sec) = {round(transfer_rate_bytes_per_sec / 1e9, 0)}")
print(f"\n\nRESULTS:\n\tTotal FLOPS = {layer_flops:.2e}\n\tTotal Bytes Read = {layer_bytes_read:.2e}\n\tLayer Size Bytes = {layer_size_bytes:.2e}\n\tUnique Bytes Read = {unique_bytes_read:.2e}\n\n\tAvg. Tok per Exp = {round(avg_m_per_expert, 0)}\n\tRouted Exp Intensity: {routed_exp_intensity}\n\n\tOverall Layer Processing Intensity = {overall_intensity:.3f}\n\tTemporally-Amoritized Intensity (accounting for just unique bytes) = {temporal_intensity:.3f}\n\n\tLayer Compute Time (ms) = {layer_compute_time * 1e3:.3f}\n\tLayer Transfer Time (ms) = {layer_transfer_time * 1e3:.3f}\n")
print(f"Layer-Wise Difference (compute - transfer, in ms) = {(layer_compute_time - layer_transfer_time) * 1e3:.3f}\n")


