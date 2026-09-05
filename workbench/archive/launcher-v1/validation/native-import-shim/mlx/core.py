"""Source-only launcher tests; no tensor or transport operations are provided."""

class cuda:
    @staticmethod
    def is_available():
        return False
