import argparse
def parse():
    p = argparse.ArgumentParser("HMGNN", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--cuda', type=str, default='0', help='gpu')
    p.add_argument('--city', type=str, default='IST', help='dataset name (e.g.NYC,TKY,IST,SP,KL,JK)')
    p.add_argument('--epochs', type=int, default=5010, help='number of epochs to train')
    p.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    p.add_argument('--multihead', type=int, default=5, help='number of multiheads')
    p.add_argument('--lambda_1', type=int, default=1, help='number of lambda_1')
    p.add_argument('--lambda_2', type=int, default=2, help='number of lambda_2')
    p.add_argument('--lambda_3', type=int, default=3, help='number of lambda_3')

    # 新增的对抗训练参数
    p.add_argument('--adversarial', type=str, choices=['none', 'random', 'fgsm', 'pgd'], default='random',
                        help='Type of adversarial training')
    p.add_argument('--epsilon', type=float, default=1e-3, help='Perturbation magnitude')
    p.add_argument('--alpha', type=float, default=1e-4, help='Step size for PGD')
    p.add_argument('--pgd_iters', type=int, default=10, help='Number of iterations for PGD')


    args = p.parse_args()
    return args