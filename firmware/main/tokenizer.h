#pragma once

#include <stddef.h>
#include <stdint.h>

typedef struct {
    int vocab_size;
    int max_token_length;
    char **vocab;
    float *scores;
    char *pool;
} tokenizer_t;

// Parse a llama2.c tokenizer.bin image already resident in memory.
int tok_parse(tokenizer_t *t, const uint8_t *blob, size_t size, int vocab_size);

// Free everything tok_parse() allocated and zero the struct.
void tok_free(tokenizer_t *t);

// Encode text to token ids. Returns the number of tokens written.
int tok_encode(const tokenizer_t *t, const char *text, int bos, int *out, int cap);

// Decode one token into buf (NUL terminated). Returns bytes written.
int tok_decode(const tokenizer_t *t, int prev, int token, char *buf, int cap);
