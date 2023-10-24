

if __name__ == '__main__':   
    import sys
    import os

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))



    from ml_logger import logger, instr, needs_relaunch
    from analysis import RUN
    from ml_logger import RUN
    import jaynes
    from scripts.train import main
    from config.locomotion_config import Config
    from params_proto.neo_hyper import Sweep


    sweep = Sweep(RUN, Config).load("default_inv.jsonl")

    for kwargs in sweep:
        logger.print(RUN.prefix, color='green')
        jaynes.config("local")
        thunk = instr(main, **kwargs)
        jaynes.run(thunk)

    jaynes.listen()
