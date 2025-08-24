project_name = "EmotionDetection"

BATCH_SIZE = 4
VAL_BATCH_SIZE = 2
MAX_EPOCHS = 4
LR = 2e-4
MODEL_NAME = "memocmt"
BERT_MODEL = "FacebookAI/roberta-base"
W2V_MODEL = "facebook/wav2vec2-base"
MAX_TEXT_LEN = 128
WAV_SAMPLING_RATE = 16000
FREEZE_ENCODER = True
SEED = 42
NUM_WORKERS = 2
OUTPUT_DIR = "saved_models"
CHECKPOINT = None
MODEL_PARAMS = {}
WARMUP_FACTOR = 0.1

GRADIENT_ACCUMULATION_STEPS = 1  # not implemented


wandb_mode = "online"
save_strategy = "epoch"
log_level = "debug"
run_name = f"{MODEL_NAME}_bs-{BATCH_SIZE}"
LOG_INTERVAL = 5


class MemoCMT:
    text_encoder_dim = 768
    audio_encoder_dim = 768
    fusion_dim = 768
    num_attention_head = 8
    linear_layer_output = [128]
    fusion_head_output_type = "cls"
    dropout = 0.1
