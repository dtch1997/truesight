import pathlib 

from third_party.question import Question

curr_dir = pathlib.Path(__file__).parent

QUESTION = Question.from_yaml("blue_or_red", curr_dir)