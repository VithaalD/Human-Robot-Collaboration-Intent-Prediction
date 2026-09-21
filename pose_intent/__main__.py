"""Run from the repository root: python -m pose_intent --help."""
import argparse
import json
from pathlib import Path
import torch
from .features import InputConfig


def main():
    parser = argparse.ArgumentParser(description='DeepLabCut movement and RGB intent pipeline')
    commands = parser.add_subparsers(dest='command', required=True)
    extract = commands.add_parser('extract', help='Run trained DLC on a full video and save coordinates/provenance')
    extract.add_argument('--video', required=True)
    extract.add_argument('--output', required=True)
    prepare = commands.add_parser('prepare', help='Build synchronized windows from reviewed segment annotations')
    prepare.add_argument('--manifest', required=True)
    prepare.add_argument('--output', required=True)
    prepare.add_argument('--input-config', help='JSON InputConfig overrides; saved with every checkpoint')
    prepare.add_argument('--stride', type=float, default=0.2)
    prepare.add_argument('--rgb-weights', help='Legacy RGB weights for frozen features and baseline')
    train = commands.add_parser('train', help='Train motion or fusion; select best validation macro F1')
    train.add_argument('--data', required=True)
    train.add_argument('--output', required=True)
    train.add_argument('--modality', choices=['motion', 'fusion'], default='motion')
    train.add_argument('--epochs', type=int, default=30)
    train.add_argument('--batch-size', type=int, default=32)
    train.add_argument('--lr', type=float, default=0.001)
    train.add_argument('--seed', type=int, default=42)
    evaluate = commands.add_parser('evaluate', help='Evaluate held-out windows and interaction timing')
    evaluate.add_argument('--data', required=True)
    evaluate.add_argument('--checkpoint', required=True)
    evaluate.add_argument('--split', choices=['val', 'test'], default='test')
    evaluate.add_argument('--output', required=True)
    replay = commands.add_parser('replay', help='Predict from recorded video and matching exported pose')
    replay.add_argument('--checkpoint', required=True)
    replay.add_argument('--video', required=True)
    replay.add_argument('--metadata', required=True)
    replay.add_argument('--pose', required=True)
    replay.add_argument('--output', required=True)
    replay.add_argument('--rgb-weights')
    live = commands.add_parser('live', help='Open camera with timestamp-aligned DLC and intent prediction')
    live.add_argument('--checkpoint', required=True)
    live.add_argument('--rgb-weights')
    live.add_argument('--camera', type=int, default=0)
    live.add_argument('--output', help='Optional new JSONL log path')
    for command in (extract, live):
        command.add_argument('--project', required=True, help='DLC project config.yaml')
        command.add_argument('--model-config', required=True, help='Trained DLC pytorch_config.yaml')
        command.add_argument('--snapshot', required=True, help='Trained DLC snapshot .pt')
    for command in (extract, prepare, train, evaluate, replay, live):
        command.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.command == 'extract':
        from .extraction import extract
        result = extract(args.video, args.project, args.model_config, args.snapshot, args.output, args.device)
    elif args.command == 'prepare':
        from .data import prepare
        config = json.loads(Path(args.input_config).read_text()) if args.input_config else {}
        result = prepare(args.manifest, args.output, InputConfig(**config), args.stride, args.rgb_weights, args.device)
    elif args.command == 'train':
        from .training import train
        result = train(args.data, args.output, args.modality, args.epochs, args.batch_size, args.lr, args.seed, args.device)
    elif args.command == 'evaluate':
        from .training import evaluate
        destination = Path(args.output)
        if destination.exists():
            raise ValueError('Choose a new report path')
        result, rows = evaluate(args.data, args.checkpoint, args.split, args.device)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('x', encoding='utf-8') as stream:
            json.dump(dict(summary=result, predictions=rows), stream, indent=2)
    elif args.command == 'replay':
        from .replay import replay
        result = replay(args.checkpoint, args.video, args.metadata, args.pose, args.output, args.rgb_weights, args.device)
    else:
        from .runtime import live
        live(args.checkpoint, args.project, args.model_config, args.snapshot, args.rgb_weights,
             args.camera, args.device, args.output)
        result = {'status': 'closed'}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
