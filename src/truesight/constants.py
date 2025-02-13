import sys  
from truesight.utils.path import get_current_dir

PROJECT_ROOT = get_current_dir(__file__).parent.parent.resolve().absolute()
sys.path.append(str(PROJECT_ROOT))