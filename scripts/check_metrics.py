import torch, glob
for pattern in ['checkpoints/E_phase1_rebalance_l4_*/best_model.pt',
                'checkpoints/E_phase1_rebalance_a100_*/best_model.pt']:
    for f in sorted(glob.glob(pattern)):
        tag = 'L4' if 'l4' in f else 'A100'
        c = torch.load(f, map_location='cpu', weights_only=False)
        ep = c.get('epoch', '?')
        dice = c.get('best_dice', 0)
        pq = c.get('best_pq', 0)
        print(f'{tag}: epoch={ep}, best_dice={dice:.4f}, best_pq={pq:.4f}')
