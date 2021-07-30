import confuse


def load_config(config: confuse.Configuration, filename: str):
    config.set_file(filename)
