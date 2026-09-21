"""Train only on train; select checkpoints on validation; test is explicit."""
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import WindowDataset
from .features import LABELS
from .models import IntentModel, RGB_CONFIG, load_intent


def forward(model, batch, device):
    return model(batch['features'].float().to(device), batch['mask'].float().to(device),
                 batch['likelihood'].float().to(device), batch.get('rgb', None).float().to(device)
                 if 'rgb' in batch else None)


def classification_metrics(truth, predictions):
    truth, predictions = np.asarray(truth), np.asarray(predictions)
    matrix = np.zeros((3, 3), dtype=int)
    for actual, pred in zip(truth, predictions):
        matrix[int(actual), int(pred)] += 1
    precision = np.diag(matrix) / np.maximum(matrix.sum(0), 1)
    recall = np.diag(matrix) / np.maximum(matrix.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return dict(class_order=list(LABELS), confusion_matrix=matrix.tolist(),
                accuracy=float((truth == predictions).mean()), macro_f1=float(f1.mean()),
                recall=recall.tolist(), precision=precision.tolist(),
                missed_interaction_window_rate=float((predictions[truth == 0] != 0).mean()) if np.any(truth == 0) else None,
                false_interaction_window_rate=float((predictions[truth != 0] == 0).mean()) if np.any(truth != 0) else None)


def predict_dataset(model, dataset, device, batch_size=32):
    truths, probs = [], []
    model.eval()
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            logits = forward(model, batch, device)
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite model output')
            probs.extend(logits.softmax(-1).cpu().tolist())
            truths.extend(batch['label'].tolist())
    return np.asarray(truths), np.asarray(probs)


def train(folder, output, modality='motion', epochs=30, batch_size=32, lr=0.001, seed=42, device='cpu'):
    if epochs < 1 or batch_size < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError('Use positive epochs, batch size and learning rate')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    training, validation = WindowDataset(folder, 'train'), WindowDataset(folder, 'val')
    for dataset in (training, validation):
        if set(r['label'] for r in dataset.records) != set(LABELS):
            raise ValueError('Train and validation must each include all three classes')
    if modality == 'fusion' and not training.index['rgb_sha256']:
        raise ValueError('Prepare with --rgb-weights before training fusion')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Choose an empty training output directory')
    model = IntentModel(modality).to(device)
    # Only observed TRAIN features contribute to normalization.
    total = torch.zeros_like(model.feature_mean, device='cpu')
    square, count = total.clone(), total.clone()
    for batch in DataLoader(training, batch_size=batch_size):
        f, m = batch['features'], batch['mask']
        total += (f * m).sum((0, 1))
        square += (f.square() * m).sum((0, 1))
        count += m.sum((0, 1))
    mean = total / count.clamp_min(1)
    std = (square / count.clamp_min(1) - mean.square()).clamp_min(0).sqrt().clamp_min(0.01)
    model.feature_mean.copy_(mean.to(device))
    model.feature_std.copy_(std.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    class_counts = np.bincount([LABELS.index(r['label']) for r in training.records], minlength=3)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(len(training) / (3 * class_counts), dtype=torch.float32, device=device))
    history, best = [], -1
    loader = DataLoader(training, batch_size=batch_size, shuffle=True, num_workers=0)
    for epoch in range(epochs):
        model.train()
        loss_sum = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(forward(model, batch, device), batch['label'].long().to(device))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * len(batch['label'])
        truth, probabilities = predict_dataset(model, validation, device, batch_size)
        metrics = classification_metrics(truth, probabilities.argmax(-1))
        row = dict(epoch=epoch + 1, loss=loss_sum / len(training), validation=metrics)
        history.append(row)
        print(json.dumps(row), flush=True)
        if metrics['macro_f1'] > best:
            best = metrics['macro_f1']
            checkpoint = dict(format='pose-intent-v1', input_config=training.cfg.dictionary(),
                              pose_identity=training.index['pose_identity'],
                              rgb_config=RGB_CONFIG, rgb_sha256=training.index['rgb_sha256'],
                              modality=modality, hidden=model.hidden,
                              state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                              train_sessions=sorted({r['session_id'] for r in training.records}),
                              validation_sessions=sorted({r['session_id'] for r in validation.records}),
                              development_participants=sorted({r['participant_id'] for r in training.records + validation.records}),
                              group_participants=training.index['group_participants'],
                              manifest_sha256=training.index['manifest_sha256'], seed=seed, epoch=epoch + 1)
            torch.save(checkpoint, output / 'best.pt')
        (output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    return dict(checkpoint=str(output / 'best.pt'), validation_macro_f1=best)


def evaluate(folder, checkpoint_path, split='test', device='cpu'):
    data = WindowDataset(folder, split)
    model, cfg, saved = load_intent(checkpoint_path, device)
    if json.dumps(cfg.dictionary(), sort_keys=True) != json.dumps(data.cfg.dictionary(), sort_keys=True):
        raise ValueError('Dataset input configuration differs from checkpoint')
    if saved['modality'] == 'fusion' and saved['rgb_sha256'] != data.index['rgb_sha256']:
        raise ValueError('RGB feature checkpoint differs from training')
    if saved['pose_identity'] != data.index['pose_identity']:
        raise ValueError('Pose extractor differs from training')
    if split == 'test':
        seen = set(saved['train_sessions'] + saved['validation_sessions'])
        if any(r['session_id'] in seen for r in data.records):
            raise ValueError('Test session was used during model development')
        if saved['group_participants'] and any(r['participant_id'] in saved['development_participants'] for r in data.records):
            raise ValueError('Test participant was used during model development')
    truth, probabilities = predict_dataset(model, data, device)
    predicted = probabilities.argmax(-1)
    report = dict(model=saved['modality'], split=split, windows=len(data),
                  metrics=classification_metrics(truth, predicted),
                  preparation_rejections=data.index['rejection_counts'])
    if data.index['rgb_sha256']:
        baseline = np.array([data[i]['rgb_logits'].numpy().argmax() for i in range(len(data))])
        report['rgb_baseline_same_windows'] = classification_metrics(truth, baseline)
    rows = [dict(r, predicted=LABELS[int(p)], probabilities=prob.tolist())
            for r, p, prob in zip(data.records, predicted, probabilities)]
    events = []
    for take in data.index['takes']:
        if take['split'] != split:
            continue
        take_rows = [r for r in rows if r['take_id'] == take['take_id']]
        for segment in take['segments']:
            if segment['label'] != 'INTERACTION':
                continue
            # Report observed detection delay, with a fixed 1-second anticipation interval.
            hits = [r['end'] for r in take_rows if r['predicted'] == 'INTERACTION'
                    and max(0, segment['start'] - 1) <= r['end'] < segment['end']]
            events.append(dict(take_id=take['take_id'], onset=segment['start'],
                               first_detection_offset_seconds=min(hits) - segment['start'] if hits else None))
    report['interaction_events'] = events
    report['event_note'] = 'Negative offset means early detection; excludes compute latency. Window false alarms must also be reviewed. Rejected windows are unavailable, not correct predictions.'
    return report, rows
