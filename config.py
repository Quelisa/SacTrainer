import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', required=True, help="model directory")
parser.add_argument('--data_path', default='./data', help="directory containing the data")
parser.add_argument('--mode', required=True, help="task type(classify or retrieve or mlm")
parser.add_argument('--do_train', default=False, help="train model")
parser.add_argument('--do_eval', default=False, help="eval model")
parser.add_argument('--output_dir', default='./result', help="directory containing the best model and result")


parser.add_argument('--fp16', default=False, type=bool, help="use fp16")
parser.add_argument('--epoch_num', default=3, type=int, help="num of epoch")
parser.add_argument('--batch_size', default=64, type=int,  help="batch size")
parser.add_argument('--max_len', default=32, type=int, help="max sequence length")
parser.add_argument('--learning_rate', default=5e-5, type=float, help="learning rate")
parser.add_argument('--clip_max_grad_norm', default=2, type=int, help="")
parser.add_argument('--gradient_accumulation_steps', default=4, type=int, help="")


args = parser.parse_args()
