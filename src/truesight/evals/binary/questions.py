from .binary import BinaryQuestionConfig, BinaryQuestion

concepts = [
    ("smooth", "sharp"),
    ("blue", "red"),
    ("good", "bad"),
    ("happy", "sad"),
    ("open", "closed"),
    ("hot", "cold"),
    ("fast", "slow"),
    ("easy", "hard"),
    ("bright", "dark"),
    ("high", "low"),
]

CONFIGS = [
    BinaryQuestionConfig(first=first, second=second)
    for (first, second) in concepts
]

QUESTIONS = {
    config.id: BinaryQuestion(config)
    for config in CONFIGS
}

