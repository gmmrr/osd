STEP_1_AUDIO_SAMPLE_RATE = 16000  # standardization (not recommended to adjust)

STEP_2_NORMALIZE_TARGET_DBFS = -20.0  # normalization target loudness
STEP_2_NORMALIZE_LIMIT_DB = 24.0  # maximum allowed gain

STEP_3_VAD_SMOOTHING_WINDOW = 400  # smoothing window for energy expansion (not recommended to adjust)
STEP_3_VAD_ENERGY_REL_THRESHOLD = 0.3  # expand detected speech boundaries using relative energy
STEP_3_VAD_EXPAND_PRE = 15  # boundary expansion before each VAD segment, in ms
STEP_3_VAD_EXPAND_POST = 15  # boundary expansion after each VAD segment, in ms
STEP_3_VAD_EXPAND_DELTA = 5  # step size when expanding VAD boundaries, in ms
STEP_3_VAD_PAD = 50  # extra padding around each VAD segment, in ms
STEP_3_VAD_THRESHOLD = 0.05  # relative RMS threshold for active segment detection
STEP_3_VAD_MIN_SPEECH = 25  # minimum accepted speech segment duration, in ms
STEP_3_VAD_MIN_SILENCE = 75  # minimum silence gap used to merge neighboring segments, in ms
STEP_3_VAD_RMS_FRAME = 25  # RMS frame length, in ms
STEP_3_VAD_RMS_HOP = 10  # RMS hop size, in ms

STEP_4_PRE_SILENCE = 50  # silence padding before each mixture, in ms
STEP_4_POST_SILENCE = 50  # silence padding after each mixture, in ms
STEP_4_SAME_SPEAKER_GAP = 50  # minimum gap between segments from the same speaker, in msㄋ
STEP_4_MAX_SPEAKERS = 3  # maximum speakers per mixture
STEP_4_MAX_SPEAKERS_PER_FRAME = 2  # maximum overlapping speakers per frame
STEP_4_RATIO_TRIES = 8  # random sampling attempts per overlap fit
STEP_4_GAIN_DB_RANGE = (-3.0, 3.0)  # per-source gain range in dB (not recommended to adjust)
STEP_4_MAX_BUILD_TRIALS = 10000  # maximum metadata construction attempts
STEP_4_MIN_SEGMENT_DURATION = 500  # minimum active segment duration, in ms
STEP_4_MIN_MIXTURE_DURATION = 15000  # minimum mixture duration, in ms
STEP_4_MAX_MIXTURE_DURATION = 25000  # maximum mixture duration, in ms
STEP_4_OVERLAP_RATIO = 0.2  # overlap ratio center
STEP_4_OVERLAP_RANDOM_OFFSET = 0.05  # overlap ratio random offset
STEP_4_SEED = 42  # random seed for reproducible metadata generation (not recommended to adjust)

STEP_5_TRAIN_RATIO = 0.8  # dataset split ratios
STEP_5_DEV_RATIO = 0.1  # dataset split ratios
STEP_5_TEST_RATIO = 0.1  # dataset split ratios