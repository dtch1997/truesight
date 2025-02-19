import sys  
import os
from dotenv import load_dotenv
from truesight.utils.path import get_current_dir

# set the project root
PROJECT_ROOT = get_current_dir(__file__).parent.parent.resolve().absolute()
sys.path.append(str(PROJECT_ROOT))

# set the cache dir
OPENAI_FINETUNER_CACHE_DIR = PROJECT_ROOT / ".cache" / "openai_finetuner"
os.environ["OPENAI_FINETUNER_CACHE_DIR"] = str(OPENAI_FINETUNER_CACHE_DIR)

QUESTION_RESULTS_CACHE_DIR = PROJECT_ROOT / ".cache" / "question_results"
os.environ["QUESTION_RESULTS_CACHE_DIR"] = str(QUESTION_RESULTS_CACHE_DIR)

# load the environment variables
load_dotenv(PROJECT_ROOT / ".env")

# check API keys enabled
assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set"