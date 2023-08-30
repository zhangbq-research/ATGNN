
import os
print('pid:', os.getpid())

from time import time
from parser_ensemble import get_args
from chem_lib.models import EnsembleContextAwareRelationNet, Meta_Trainer
from chem_lib.utils import count_model_params
import random
import numpy as np
import torch

def seed_torch(seed=21):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.deterministic = True  #cpu/gpu结果一致
    torch.backends.cudnn.benchmark = False   #训练集变化不大时使训练加速

def main():
    root_dir = '.'
    args = get_args(root_dir)
    seed_torch(7890)
    model = EnsembleContextAwareRelationNet(args)
    count_model_params(model)
    model = model.to(args.device)
    trainer = Meta_Trainer(args, model)

    t1=time()
    print('Initial Evaluation')
    best_avg_auc=0
    best_avg_acc=0
    for epoch in range(1, args.epochs + 1):
        print('----------------- Epoch:', epoch,' -----------y------')
        trainer.train_step()

        if epoch % args.eval_steps == 0 or epoch==1 or epoch==args.epochs:
            print('Evaluation on epoch',epoch)
            if args.tentimes:
                best_avg_acc, best_avg_acc_ci95 = trainer.test_step()
            else:
                best_avg_auc, best_avg_acc_ci95 = trainer.test_step()

        if epoch % args.save_steps == 0:
            trainer.save_model()
        print('Time cost (min):', round((time()-t1)/60,3))
        t1=time()

    print('Train done.')
    print('Best Avg AUC:',best_avg_auc)
    print('Best Avg ACC: {} ± {}'.format(best_avg_acc, best_avg_acc_ci95))

    trainer.conclude()

    if args.save_logs:
        trainer.save_result_log()

if __name__ == "__main__":
    main()
