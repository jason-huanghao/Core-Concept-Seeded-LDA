import sys

module_path = r"../pybind_module"
sys.path.append(module_path)
import LDAs


lda = LDAs.LDA
zlabel = LDAs.Zlabel
seed_lda = LDAs.SeededLDA


def load_lda_module(pybind_module_dir):
    '''
    :param module_dir_path: the pybind module of all LDAs
    :return:
    '''
    import sys
    import os

    global lda, zlabel, seed_lda

    if not os.path.exists(pybind_module_dir + "/Seed_WE_LDA.pyd"):
        print("LDA modules are not compiled!")
        exit(0)
    sys.path.append(pybind_module_dir)

    import LDAs

    lda = LDAs.LDA
    zlabel = LDAs.Zlabel
    seed_lda = LDAs.SeededLDA



