"""SPARK configuration and boundary-only projector provenance.

No tensor reads in the round scheduler or host metric calculations. Projector
copies/hashes run only at initialization, checkpoint/resume and termination.
"""
import hashlib
import torch

UPDATE_INTERVALS = (0, 1, 5, 10, 15)
HOST_COUNTER_NAMES = ('opd_feedback_rounds', 'opd_skipped_rounds')
CONFIG_FIELDS = ('opd_ablation_name', 'opd_projector_init', 'opd_projector_mode',
                 'opd_projector_seed', 'opd_train_projector', 'opd_update_interval_rounds')
PROVENANCE_FIELDS = ('opd_initial_a_sha256', 'opd_final_a_sha256', 'opd_a_change_norm')


def feedback_due(round_number, interval):
    """One-based verification batch round, reset by each generate() call."""
    return interval > 0 and (round_number - 1) % interval == 0


def ablation_config(args):
    train = str(args.opd_train_projector).lower() in ('1', 'true')
    interval = int(args.opd_update_interval_rounds)
    init = args.opd_projector_init
    name = args.opd_ablation_name
    if not name:
        name = ('learned_random' if train else 'frozen_random') if init == 'random_orthogonal' else ('main' if train else 'frozen_head_aligned')
        if interval != 1:
            name += f'_interval{interval}'
    return dict(opd_ablation_name=name, opd_projector_init=init,
                opd_projector_seed=int(args.opd_projector_seed),
                opd_train_projector=train, opd_update_interval_rounds=interval,
                opd_projector_mode=('inactive' if interval == 0 else 'learned' if train else 'frozen'),
                opd_rank=int(args.opd_rank), opd_topk=int(args.opd_topk),
                opd_fast_lr=float(args.opd_fast_lr), opd_projector_lr=args.opd_projector_lr,
                opd_visited_weight=float(args.opd_visited_weight),
                opd_frontier_weight=float(args.opd_frontier_weight),
                opd_state_selection='visited_and_expanded_one_hop_frontier')


def validate_resume_config(saved, current):
    if current is None:
        return
    if saved is None:
        # Legacy runs used head alignment, learned A, and feedback every round.
        if (current['opd_projector_init'] != 'head_aligned' or
                current['opd_update_interval_rounds'] != 1 or
                not current['opd_train_projector'] or current['opd_projector_seed'] != 42):
            raise ValueError('resume ablation configuration missing in legacy checkpoint')
        return
    differences = [key for key in current if saved.get(key) != current[key]]
    if differences:
        raise ValueError('resume ablation configuration mismatch: ' + ', '.join(differences))


def tensor_sha256(value):
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def initialize_provenance(model):
    value = model.opd_projector.detach().float().cpu().clone()
    model._opd_initial_projector = value
    model._opd_initial_a_sha256 = tensor_sha256(value)


def final_provenance(model):
    if getattr(model, 'opd_projector', None) is None:
        return {}
    value = model.opd_projector.detach().float().cpu()
    initial = model._opd_initial_projector
    return dict(opd_initial_a_sha256=model._opd_initial_a_sha256,
                opd_final_a_sha256=tensor_sha256(value),
                opd_a_change_norm=float(torch.linalg.vector_norm(value - initial)))


def round_metrics(counters):
    """All inputs are existing aggregated Python counters, never tensors."""
    rounds = counters.get('verification_batches', 0.)
    feedback = counters.get('opd_feedback_rounds', 0.)
    updates = counters.get('opd_updates', 0.)
    return dict(total_verification_rounds=rounds, opd_feedback_rounds=feedback,
                opd_skipped_rounds=counters.get('opd_skipped_rounds', 0.),
                opd_actual_b_updates=updates,
                opd_actual_update_frequency=updates / rounds if rounds > 0 else 0.,
                opd_feedback_frequency=feedback / rounds if rounds > 0 else 0.)
