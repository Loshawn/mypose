import os
import time
import sys
import torch
import matplotlib.pyplot as plt


class Logger(object):

    def __init__(self, args, rank):
        """Create a summary writer logging to log_dir."""
        self.args = args
        self.dir_root = args.dir_root

        self.info = {"Epoch": 0, "train_loss": [], "test_loss": [], "acc": []}
        self.rank = rank

        if rank == 0:
            self._make_dir()

            with open(self.dir_id + "/config.txt", "w") as f:
                f.write(time.strftime('%Y-%m-%d-%H-%M') + "\n\n")
                f.write('==> torch version: {}\n'.format(torch.__version__))
                f.write('==> cudnn version: {}\n'.format(
                    torch.backends.cudnn.version()))
                f.write('==> Cmd:\n')
                f.write(str(sys.argv))
                f.write('\n==> args:\n')

                for arg in vars(args):
                    f.write(f"{arg}: {getattr(args, arg)} \n")
                f.write("\n")

            self.log_file = open(self.dir_id + '/log.txt', "w")

    def _make_dir(self):
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        self.dir_id = os.path.join(self.dir_root, "exp", self.args.save,
                                   f"log_{time_str}")
        self.dir_model = os.path.join(self.dir_id, "model")
        self.dir_result = os.path.join(self.dir_id, "result")

        for path in [self.dir_id, self.dir_model, self.dir_result]:
            if not os.path.exists(path):
                os.makedirs(path)

    def add_log(self, log):
        if self.rank == 0:
            self.log_file.write(log + '\n')
            self.log_file.flush()

    def draw_loss(self):
        if self.rank == 0:
            epoch = range(1, self.info["Epoch"] + 2)
            plt.figure()
            plt.plot(epoch, self.info["train_loss"], label="Train loss")
            plt.plot(epoch, self.info["test_loss"], label="Test loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.title(f"Loss")
            plt.grid()
            plt.savefig(f"{self.dir_result}/Loss.png", dpi=1000)
            plt.close()

            plt.figure()
            plt.plot(epoch, self.info["acc"], label="Acc")
            plt.xlabel("Epoch")
            plt.ylabel("Acc")
            plt.legend()
            plt.title(f"Acc")
            plt.grid()
            plt.savefig(f"{self.dir_result}/Acc.png", dpi=1000)
            plt.close()

    def printf(self, string):
        if self.rank == 0:
            print(string)
