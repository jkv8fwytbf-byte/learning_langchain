from importlib import import_module

demo_basic_chain = import_module("1_basic_chain").demo_basic_chain
demo_parallel_chain = import_module("2_parallel_chain").demo_parallel_chain
demo_passthrough_chain = import_module("3_passthrough_chain").demo_passthrough_chain
demo_chain_branching = import_module("4_chain_branching").demo_chain_branching
demo_debbuging = import_module("5_debugging").demo_debbuging

if __name__ == "__main__":
    # demo_basic_chain()
    # demo_parallel_chain()
    # demo_passthrough_chain()
    # demo_chain_branching()
    demo_debbuging()
