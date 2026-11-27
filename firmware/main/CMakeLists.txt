idf_component_register(
    SRCS "main.c" "llm.c" "tokenizer.c" "model_lib.c" "model_prefs.c" "moe.c"
         "expert_access_bench.c" "dot_s8.S"
    INCLUDE_DIRS "."
)
