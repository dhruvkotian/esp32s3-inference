// llama2.c tokenizer (SentencePiece BPE with byte fallback), ported from run.c.
//
// File layout: int32 max_token_length, then vocab_size entries of
// (float32 score, int32 length, bytes). Byte tokens live at ids 3..258
// because the first three entries are <unk>, <s>, </s>, hence the +3
// fallback offset used by run.c and reproduced here.

#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int tok_parse(tokenizer_t *t, const uint8_t *blob, size_t size, int vocab_size)
{
    memset(t, 0, sizeof(*t));
    if (size < 8 || vocab_size <= 0) {
        return -1;
    }
    memcpy(&t->max_token_length, blob, 4);
    t->vocab_size = vocab_size;
    t->vocab = calloc((size_t)vocab_size, sizeof(char *));
    t->scores = calloc((size_t)vocab_size, sizeof(float));
    t->pool = malloc(size + (size_t)vocab_size + 1);
    if (!t->vocab || !t->scores || !t->pool) {
        return -2;
    }

    size_t off = 4;
    size_t poff = 0;
    for (int i = 0; i < vocab_size; i++) {
        if (off + 8 > size) {
            return -3;
        }
        float score;
        int32_t len;
        memcpy(&score, blob + off, 4);
        off += 4;
        memcpy(&len, blob + off, 4);
        off += 4;
        if (len < 0 || off + (size_t)len > size) {
            return -3;
        }
        t->scores[i] = score;
        memcpy(t->pool + poff, blob + off, (size_t)len);
        t->pool[poff + len] = '\0';
        t->vocab[i] = t->pool + poff;
        poff += (size_t)len + 1;
        off += (size_t)len;
    }
    return 0;
}

static int str_lookup(const tokenizer_t *t, const char *s)
{
    for (int i = 0; i < t->vocab_size; i++) {
        if (strcmp(t->vocab[i], s) == 0) {
            return i;
        }
    }
    return -1;
}

int tok_encode(const tokenizer_t *t, const char *text, int bos, int *out, int cap)
{
    if (!t || !text || !out) {
        return -1;
    }
    int n = 0;
    if (bos && n < cap) {
        out[n++] = 1;
    }
    if (text[0] != '\0' && n < cap) {
        int dummy = str_lookup(t, " ");
        if (dummy >= 0) {
            out[n++] = dummy;
        }
    }

    size_t buf_cap = (size_t)t->max_token_length * 2 + 4;
    char *buf = malloc(buf_cap);
    if (!buf) {
        return -1;
    }
    size_t buf_len = 0;

    for (const char *c = text; *c != '\0'; c++) {
        if (((unsigned char)*c & 0xC0) != 0x80) {
            buf_len = 0;
        }
        if (buf_len + 1 >= buf_cap) {
            buf_len = 0;
        }
        buf[buf_len++] = *c;
        buf[buf_len] = '\0';

        if (((unsigned char)*(c + 1) & 0xC0) == 0x80 && buf_len < 4) {
            continue;
        }

        int id = str_lookup(t, buf);
        if (id >= 0) {
            if (n < cap) {
                out[n++] = id;
            }
        } else {
            for (size_t i = 0; i < buf_len; i++) {
                if (n < cap) {
                    out[n++] = (unsigned char)buf[i] + 3;
                }
            }
        }
        buf_len = 0;
    }
    free(buf);

    // Merge the best consecutive pair each round, highest score first.
    char *cand = malloc(buf_cap);
    if (!cand) {
        return n;
    }
    while (1) {
        float best_score = -1e10f;
        int best_id = -1;
        int best_idx = -1;
        for (int i = 0; i < n - 1; i++) {
            snprintf(cand, buf_cap, "%s%s", t->vocab[out[i]], t->vocab[out[i + 1]]);
            int id = str_lookup(t, cand);
            if (id >= 0 && t->scores[id] > best_score) {
                best_score = t->scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        if (best_idx < 0) {
            break;
        }
        out[best_idx] = best_id;
        for (int i = best_idx + 1; i < n - 1; i++) {
            out[i] = out[i + 1];
        }
        n--;
    }
    free(cand);
    return n;
}

int tok_decode(const tokenizer_t *t, int prev, int token, char *buf, int cap)
{
    if (!t || !buf || cap < 1 || token < 0 || token >= t->vocab_size) {
        if (buf && cap > 0) {
            buf[0] = '\0';
        }
        return 0;
    }
    const char *p = t->vocab[token];
    if (prev == 1 && p[0] == ' ') {
        p++;
    }
    unsigned int byte_val;
    if (strlen(p) == 6 && p[0] == '<' && p[1] == '0' && p[2] == 'x' && p[5] == '>' &&
        sscanf(p, "<0x%02X>", &byte_val) == 1) {
        buf[0] = (char)byte_val;
        buf[1] = '\0';
        return 1;
    }
    int n = (int)strlen(p);
    if (n > cap - 1) {
        n = cap - 1;
    }
    memcpy(buf, p, (size_t)n);
    buf[n] = '\0';
    return n;
}
