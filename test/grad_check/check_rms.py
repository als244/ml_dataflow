import numpy as np
import torch
import os
def torch_rms(x, weights, eps):
	x_float = x.float()
	x_norm_out = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + eps)
	x_out = x_norm_out.type_as(x) * weights
	return x_out



def load_native_rms(M, N, fwd_dtype, bwd_dtype, rms_val_dtype):
	
	orig = np.fromfile("test_rms/orig_matrix.dat", dtype=fwd_dtype).reshape(M, N)
	weights = np.fromfile("test_rms/weights.dat", dtype=fwd_dtype).reshape(N)
	out = np.fromfile("test_rms/fwd_out_matrix.dat", dtype=fwd_dtype).reshape(M, N)
	rms_vals = np.fromfile("test_rms/rms_vals.dat", dtype=rms_val_dtype).reshape(M)
	
	upstream_dX = np.fromfile("test_rms/upstream_dX.dat", dtype=bwd_dtype).reshape(M, N)
	
	if os.path.exists("test_rms/resid_grad.dat"):
		resid_grad = np.fromfile("test_rms/resid_grad.dat", dtype=bwd_dtype).reshape(M, N)
	else:
		resid_grad = np.zeroes((M, N), dtype=bwd_dtype)

	return orig, weights, out, rms_vals, upstream_dX, resid_grad



M = 2048
N = 2048

eps = 1e-5

fwd_dtype = np.uint16
bwd_dtype = np.uint16
rms_val_dtype = np.float32


orig, weights, my_out, my_rms_vals, upstream_dX, resid_grad  = load_native_rms(M, N, fwd_dtype, bwd_dtype, rms_val_dtype)


torch_orig = torch.from_numpy(orig).view(torch.bfloat16).requires_grad_(True)
torch_weights = torch.from_numpy(weights).view(torch.bfloat16).requires_grad_(True)

torch_rms_out = torch_rms(torch_orig, torch_weights, eps)

torch.save(torch_rms_out, "test_rms/torch_fwd_out_matrix.pt")


## Compute Backwards derivs

torch_upstream_dX = torch.from_numpy(upstream_dX).view(torch.bfloat16)
torch_resid_grad = torch.from_numpy(resid_grad).view(torch.bfloat16)

torch_rms_out.backward(gradient=torch_upstream_dX)

torch_orig_grad = torch_orig.grad
torch.save(torch_orig_grad, "test_rms/torch_dX_matrix.pt")

torch_grad_with_resid = torch_orig_grad + torch_resid_grad
torch.save(torch_grad_with_resid, "test_rms/torch_dX_with_resid.pt")

torch_weights_grad = torch_weights.grad
torch.save(torch_weights_grad, "test_rms/torch_dWeights.pt")

