from hypothesis import settings

# No time limit per example: CI runners (Windows, macOS) are slow and uneven, and a deadline then fails at random
settings.register_profile('xtl', deadline=None, print_blob=True)
settings.load_profile('xtl')
