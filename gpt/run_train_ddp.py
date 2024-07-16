import argparse

from train import train


def run():
    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument('--data_root', required=True)
    arg_parser.add_argument('--total_batch_size', default=524288, type=int)
    arg_parser.add_argument('--batch_size', type=int, required=True)
    arg_parser.add_argument('--seq_len', type=int, required=True)
    arg_parser.add_argument('--max_steps', default=10000, type=int)
    arg_parser.add_argument('--warmup_steps', default=500, type=int)
    arg_parser.add_argument('--auto_cast_dtype', default='float16')

    arguments = arg_parser.parse_args()
    
    print(arguments)
    
    train(
        data_root=arguments.data_root, total_batch_size=arguments.total_batch_size, batch_size=arguments.batch_size,
        max_seq_len=arguments.seq_len, max_steps=arguments.max_steps, warmup_steps=arguments.warmup_steps,
        auto_cast_dtype=arguments.auto_cast_dtype)


if __name__ == '__main__':
    run()
