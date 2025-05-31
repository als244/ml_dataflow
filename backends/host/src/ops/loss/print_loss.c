#include "host_ops.h"


int print_chunk_loss_host(void * _print_chunk_loss_host_op_args){
    
    Print_Chunk_Loss_Host_Op_Args * args = (Print_Chunk_Loss_Host_Op_Args *) _print_chunk_loss_host_op_args;

    int step_num = args -> step_num;
    int round_num = args -> round_num;
    int seq_id = args -> seq_id;
    int chunk_id = args -> chunk_id;
    int num_tokens = args -> num_tokens;
    float * avg_loss_ref = args -> avg_loss_ref;

    float avg_loss = *avg_loss_ref;

    printf("[LOSS: Step %d, Round %d, Seq %d, Chunk %d, Num Tokens: %d] Avg Loss: %f\n", step_num, round_num, seq_id, chunk_id, num_tokens, avg_loss);

    return 0;
}

int print_round_loss_host(void * _print_round_loss_host_op_args){

    Print_Round_Loss_Host_Op_Args * args = (Print_Round_Loss_Host_Op_Args *) _print_round_loss_host_op_args;

    int step_num = args -> step_num;
    int round_num = args -> round_num;
    int num_seqs = args -> num_seqs;
    int num_chunks = args -> num_chunks;
    int total_tokens = args -> total_tokens;
    float * per_chunk_avg_loss = args -> per_chunk_avg_loss;

    float total_avg_chunk_loss = 0.0f;

    for (int i = 0; i < num_chunks; i++){
        total_avg_chunk_loss += per_chunk_avg_loss[i];
    }

    float avg_chunk_loss = total_avg_chunk_loss / num_chunks;

    printf("[LOSS: Step %d, Round %d, Num Seqs: %d (%d chunks), Total Tokens: %d] Avg Loss: %f\n", step_num, round_num, num_seqs, num_chunks, total_tokens, avg_chunk_loss);

    return 0;
}





