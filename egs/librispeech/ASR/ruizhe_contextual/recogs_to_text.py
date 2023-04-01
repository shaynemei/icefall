import logging
import argparse
import ast
import lhotse

logging.basicConfig(
    format = "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
    level = 10
)

def parse_opts():
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input', type=str, default=None, help='')
    parser.add_argument('--out', type=str, help='')
    parser.add_argument('--cuts', type=str, help='')

    opts = parser.parse_args()
    logging.info(f"Parameters: {vars(opts)}")
    return opts


def main(opts):
    cuts = lhotse.load_manifest(opts.cuts)

    with open(opts.input, 'r') as fin, open(opts.out, 'w') as fout:
        # lines = fin.readlines()   # read into a list
        for line in fin:
            line = line.rstrip('\n').split("\t")
            cid = line[0][:-1]
            
            if line[1][:3] == "ref":
                continue
            assert line[1][:3] == "hyp"

            # option1:
            uid = cuts[cid].supervisions[0].id

            # option2:
            # uid = cid
            # if uid not in cuts:
            #     continue

            sent = " ".join(ast.literal_eval(line[1][4:]))
            sent = sent.lower()
            print(f"{uid}\t{sent}", file=fout)


if __name__ == '__main__':
    opts = parse_opts()

    main(opts)
