# src/utils/run_name_utils.py
import random


def generate_run_name():
    """Generate a random run name like 'eloquent_octopus_042'"""
    ADJECTIVES = [
        "admiring", "adorable", "adventurous", "agile", "alert", "ambitious", "amusing",
        "balanced", "bold", "brave", "bright", "brilliant", "careful", "charming",
        "cheerful", "clever", "colorful", "courageous", "creative", "curious", "daring",
        "dazzling", "decisive", "dedicated", "determined", "diligent", "dynamic", "eager",
        "efficient", "elegant", "eloquent", "energetic", "enthusiastic", "exceptional",
        "exciting", "extraordinary", "fearless", "fierce", "focused", "friendly", "funny",
        "generous", "gentle", "glowing", "graceful", "grateful", "happy", "harmonious",
        "helpful", "honest", "hopeful", "humble", "humorous", "imaginative", "impressive",
        "independent", "innovative", "inspired", "intelligent", "intuitive", "inventive",
        "jolly", "jovial", "joyful", "keen", "kind", "knowledgeable", "lively", "loving",
        "loyal", "lucky", "magnificent", "marvelous", "motivated", "mysterious", "noble",
        "optimistic", "passionate", "patient", "peaceful", "persistent", "polite", "proud",
        "quirky", "radiant", "rational", "reliable", "remarkable", "resilient", "resourceful",
        "respected", "romantic", "sensible", "sharp", "sincere", "skilled", "smiling",
        "spirited", "splendid", "stellar", "strong", "stunning", "successful", "talented",
        "tenacious", "thoughtful", "thriving", "trustworthy", "vibrant", "vigilant",
        "vigorous", "vivid", "warm", "wise", "witty", "wonderful", "zealous"
    ]

    ANIMALS = [
        "alligator", "antelope", "armadillo", "badger", "bat", "bear", "beaver", "bee",
        "beetle", "bison", "boar", "buffalo", "butterfly", "camel", "caribou", "cat",
        "cheetah", "chicken", "chimpanzee", "chipmunk", "cobra", "coyote", "crab",
        "crane", "crocodile", "crow", "deer", "dog", "dolphin", "donkey", "dove",
        "dragonfly", "duck", "eagle", "echidna", "eel", "elephant", "elk", "falcon",
        "ferret", "finch", "fish", "flamingo", "fox", "frog", "gazelle", "gerbil",
        "giraffe", "goat", "goose", "gorilla", "grasshopper", "hamster", "hare", "hawk",
        "hedgehog", "heron", "hippo", "hornet", "horse", "hummingbird", "hyena", "ibex",
        "iguana", "impala", "jackal", "jaguar", "jellyfish", "kangaroo", "kingfisher",
        "kiwi", "koala", "kookaburra", "leopard", "lion", "lizard", "llama", "lobster",
        "lynx", "macaw", "magpie", "manatee", "mandrill", "meerkat", "mole", "mongoose",
        "monkey", "moose", "mouse", "narwhal", "newt", "nightingale", "octopus", "opossum",
        "orca", "ostrich", "otter", "owl", "panda", "panther", "parrot", "peacock",
        "pelican", "penguin", "pheasant", "porcupine", "porpoise", "python", "quail",
        "quokka", "rabbit", "raccoon", "ram", "raven", "rhinoceros", "robin", "salamander",
        "salmon", "seahorse", "seal", "shark", "sheep", "skunk", "sloth", "snake",
        "sparrow", "squirrel", "starling", "stingray", "stork", "swan", "tiger", "toad",
        "toucan", "turtle", "vulture", "walrus", "weasel", "whale", "wolf", "wolverine",
        "wombat", "woodpecker", "zebra"
    ]

    # Random number between 0-999 with leading zeros
    random_number = random.randint(0, 999)

    return f"{random.choice(ADJECTIVES)}_{random.choice(ANIMALS)}_{random_number:03d}"