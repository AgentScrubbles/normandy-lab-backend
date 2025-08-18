import argparse
from io import BytesIO
import os
import random
import string
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision
from torchvision import transforms
from PIL import Image
import face_recognition
from torch.utils.tensorboard import SummaryWriter
from PIL import Image


from base_schema import BaseSchema, all_schemas, get_schema


class Trainer:

    def __init__(self, schema: str) -> None:
        self.schema = get_schema(schema)
        self.NUM_CLASSES_PER_DIGIT: List[int] = [slider.options for slider in self.schema.get_sliders()]
        self.ALPHABET = list(string.digits + string.ascii_uppercase)

        def symbols_for(n: int) -> List[str]:
            assert 1 <= n <= len(self.ALPHABET), f"n must be in [1,{len(self.ALPHABET)}], got {n}"
            return self.ALPHABET[:n]

        # Build per-position symbol maps (char->idx and idx->char)
        self.POSITION_SYMBOLS: List[List[str]] = [symbols_for(n) for n in self.NUM_CLASSES_PER_DIGIT]
        self.CHAR2IDX: List[Dict[str, int]] = [{c: i for i, c in enumerate(sym)} for sym in self.POSITION_SYMBOLS]
        self.IDX2CHAR: List[Dict[int, str]] = [{i: c for i, c in enumerate(sym)} for sym in self.POSITION_SYMBOLS]

    def set_seed(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def make_transforms(self, img_size: int, aug: bool = True):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        if aug:
            return transforms.Compose([
                transforms.Resize(int(img_size * 1.15)),
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.04),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAutocontrast(p=0.3),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            return transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                normalize,
            ])


    @torch.no_grad()
    def eval_epoch(self, model: nn.Module, loader: DataLoader, device: torch.device, writer, epoch) -> Tuple[float, float, List[float]]:
        model.eval()
        total_loss = 0.0
        n_samples = 0
        correct_per_digit = [0 for _ in range(len(self.schema.get_sliders()))]
        exact_match = 0

        for imgs, labels, _codes in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outs = model(imgs)
            loss = 0
            preds = []
            for d, out in enumerate(outs):
                loss += F.cross_entropy(out, labels[:, d], reduction='sum')
                pred = out.argmax(dim=1)
                preds.append(pred)
                correct_per_digit[d] += (pred == labels[:, d]).sum().item()
            total_loss += loss.item()  # type: ignore
            n_samples += imgs.size(0)
            # exact match
            stacked = torch.stack(preds, dim=1)
            exact_match += (stacked == labels).all(dim=1).sum().item()

        avg_loss = total_loss / n_samples
        per_digit_acc = [c / n_samples for c in correct_per_digit]
        exact_acc = exact_match / n_samples
        sliders = self.schema.get_sliders()
        for d_idx, acc in enumerate(per_digit_acc, start=1):
            slider = sliders[d_idx - 1]
            writer.add_scalar(f"Accuracy/{slider.name}", acc, epoch)
        return avg_loss, exact_acc, per_digit_acc
    
    def train(self, args):
        log_dir = Path(args.out_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
        self.set_seed(args.seed)
        device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')

        # Datasets & loaders
        train_tf = self.make_transforms(args.img_size, aug=True)
        val_tf = self.make_transforms(args.img_size, aug=False)

        full_ds = CodeImageDataset(self.schema.get_output_dir(), transform=None, char2idx=self.CHAR2IDX, schema=self.schema)  # base without tf to share file list

        # Split
        n_total = len(full_ds)
        n_val = max(int(n_total * args.val_ratio), 1)
        n_train = n_total - n_val

        # We will recreate two datasets that share the same files but with different transforms
        indices = list(range(n_total))
        random.shuffle(indices)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        class SubsetDS(Dataset):
            def __init__(self, base: CodeImageDataset, indices: List[int], transform):
                self.base = base
                self.indices = indices
                self.transform = transform
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, i):
                img, label, code = self.base[self.indices[i]]
                img = self.transform(img)
                return img, label, code

        train_ds = SubsetDS(full_ds, train_idx, train_tf)
        val_ds = SubsetDS(full_ds, val_idx, val_tf)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.workers, pin_memory=(device.type=='cuda'))
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=(device.type=='cuda'))

        # Model
        model = CodePredictor(self.NUM_CLASSES_PER_DIGIT).to(device)

        # Optim & sched
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

        best_val = float('inf')
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_seen = 0
            for imgs, labels, _codes in train_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    outs = model(imgs)
                    loss = 0
                    for d, out in enumerate(outs):
                        loss = loss + F.cross_entropy(out, labels[:, d])
                scaler.scale(loss).backward()  # type: ignore
                scaler.step(opt)
                scaler.update()

                epoch_loss += loss.item() * imgs.size(0)  # type: ignore
                n_seen += imgs.size(0)

            sched.step()

            val_loss, exact_acc, per_digit_acc = self.eval_epoch(model, val_loader, device, writer, epoch)

            train_loss = epoch_loss / n_seen
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/exact", exact_acc, epoch)

            print(f"Epoch {epoch:02d}/{args.epochs} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | exact_acc {exact_acc:.4f}")
            # Optionally print a few per-digit accuracies
            if epoch % max(1, args.epochs // 5) == 0:
                head = ', '.join([f"d{idx+1}:{acc:.2f}" for idx, acc in enumerate(per_digit_acc[:6])])
                tail = ', '.join([f"d{idx+1}:{acc:.2f}" for idx, acc in enumerate(per_digit_acc[-6:])])
                print("  per-digit head:", head)
                print("  per-digit tail:", tail)

            # Save checkpoints
            ckpt = {
                'epoch': epoch,
                'model': model.state_dict(),
                'opt': opt.state_dict(),
                'sched': sched.state_dict(),
                'args': vars(args),
            }
            torch.save(ckpt, out_dir / f"epoch_{epoch:03d}.pt")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(ckpt, out_dir / "best.pt")
                print("  -> saved best.pt")
        writer.close()

    def infer(self, image_source: Union[str, BytesIO], ckpt_location: str, device:str = 'cpu', image_size: int = 256, verbose = False, beam_width: int = 10, topn: int = 1):
        device = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')
        ckpt = torch.load(ckpt_location, map_location=device)
        model = CodePredictor(self.NUM_CLASSES_PER_DIGIT).to(device)
        model.load_state_dict(ckpt['model'])
        tf = self.make_transforms(image_size, aug=False)
        img_base = self.load_image_for_infer(image_source, image_size, margin_ratio=0.3)
        img_base = self.resize_max_side(img_base, image_size)
        img = tf(img_base)
        probs = self.softmax_heads(model, img, device) # type: ignore

        # top-k per digit (optional print)
        if verbose:
            for pos, pr in enumerate(probs):
                topv, topi = torch.topk(pr, k=min(3, pr.numel()))
                choices = [(self.IDX2CHAR[pos][i.item()], float(v.item())) for v, i in zip(topv, topi)] # type: ignore
                print(f"Pos {pos+1:02d}: {choices}")

        # beam search for full codes
        codes = self.beam_search_codes(probs, beam_width=beam_width, topn=topn)
        resp = []
        for keypair in codes:
            resp.append(keypair[0])
        return (resp, img_base)

        # if (capture_live_dir is not None):
        #     os.makedirs(capture_live_dir, exist_ok=True)
        #     img = self.schema.capture_image(code)            if (img is not None):
        #         path = Path(infer_image)
        #         out_image_path = f"{Path(capture_live_dir)}/{path.stem}.png"
        #         overlayed_image = self.place_next_to(img, img_base)
        #         overlayed_image.save(out_image_path)

    def place_next_to(self, base_img: Image.Image, small_img: Image.Image, margin: int = 0) -> Image.Image:
        """
        Place small_img next to base_img, aligned to the bottom-right of the combined image.
        
        Args:
            base_img (Image.Image): The larger image.
            small_img (Image.Image): The smaller image to place next to it.
            margin (int): Optional space between images.
        
        Returns:
            Image.Image: New image with small_img placed next to base_img.
        """
        # Convert to RGBA to handle transparency if needed
        base = base_img.convert("RGBA")
        small = small_img.convert("RGBA")
        
        # Determine new image size (width = sum of both widths + margin, height = max of both)
        new_width = base.width + small.width + margin
        new_height = max(base.height, small.height)
        
        # Create a blank canvas
        result = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))
        
        # Paste the base image at top-left
        result.paste(base, (0, 0))
        
        # Paste the smaller image to the right of base image, bottom-aligned
        x = base.width + margin
        y = new_height - small.height
        result.paste(small, (x, y), small)
        
        return result

    def resize_max_side(self, img: Image.Image, max_side: int) -> Image.Image:
        """
        Resize an image so that the longest side is at most max_side, keeping aspect ratio.
        
        Args:
            img (Image.Image): The input image.
            max_side (int): Maximum length of the longest side (width or height).
        
        Returns:
            Image.Image: Resized image.
        """
        w, h = img.size
        # Determine scale factor
        scale = min(max_side / w, max_side / h)
        
        # Only resize if image is bigger than max_side
        if scale < 1:
            new_w = int(w * scale)
            new_h = int(h * scale)
            return img.resize((new_w, new_h))
        else:
            return img.copy()
        
    @torch.no_grad()
    def softmax_heads(self, model: nn.Module, img: torch.Tensor, device: torch.device) -> List[torch.Tensor]:
        """Returns per-digit probability tensors of shape (num_classes,) each."""
        model.eval()
        img = img.to(device)
        outs = model(img.unsqueeze(0))  # list of (1, C)
        probs = [F.softmax(o[0], dim=0) for o in outs]
        return probs


    def beam_search_codes(self, probs_per_digit: List[torch.Tensor], beam_width: int = 3, topn: int = 5) -> List[Tuple[str, float]]:
        """Combine per-digit probabilities into top-N full codes using beam search.
        Returns list of (code, prob) sorted by prob desc.
        """
        beams: List[Tuple[str, float]] = [("", 1.0)]
        for pos, probs in enumerate(probs_per_digit):
            topk = torch.topk(probs, k=min(beam_width, probs.numel()))
            next_beams = []
            for prefix, p in beams:
                for idx, p_digit in zip(topk.indices.tolist(), topk.values.tolist()):
                    ch = self.IDX2CHAR[pos][idx]
                    next_beams.append((prefix + ch, p * float(p_digit)))
            # keep the top beam_width beams
            next_beams.sort(key=lambda x: x[1], reverse=True)
            beams = next_beams[:beam_width]
        # Finally return topn from final beams (already sorted)
        return beams[:topn]


    def load_image_for_infer(self, image_source: Union[str, BytesIO], img_size: int, margin_ratio: float = 0.5) -> Image.Image:
        """
        margin_ratio: fraction of face height to extend the crop on all sides
        """
        if isinstance(image_source, str):
            # Disk path
            img = face_recognition.load_image_file(image_source)
        else:
            # File-like (BytesIO from POST)
            image_source.seek(0)  # reset stream position
            img = face_recognition.load_image_file(image_source)
        
        h_img, w_img, _ = img.shape
        
        # Detect faces
        face_locations = face_recognition.face_locations(img)
        
        if face_locations:
            # Take the first face
            top, right, bottom, left = face_locations[0]
            face_height = bottom - top
            face_width = right - left
            
            # Expand bounding box by margin_ratio
            top_exp = max(0, int(top - face_height * margin_ratio))
            bottom_exp = min(h_img, int(bottom + face_height * margin_ratio))
            left_exp = max(0, int(left - face_width * margin_ratio))
            right_exp = min(w_img, int(right + face_width * margin_ratio))
            
            face_img = img[top_exp:bottom_exp, left_exp:right_exp]
            pil_img = Image.fromarray(face_img)
        else:
            # If no face detected, fallback to original image
            pil_img = Image.fromarray(img)
        
        # Save the processed image to temp.jpg for viewing
        pil_img.save("temp.jpg")
        
        # Apply transforms
        return pil_img


class CodeImageDataset(Dataset):

    def __init__(self, root: str, char2idx, schema: BaseSchema, transform=None):
        self.root = Path(root)
        self.transform = transform
        # Collect png files only; you can extend to jpg
        self.files = [p for p in self.root.glob('*.png')]
        self.char2idx = char2idx
        self.schema = schema
        if len(self.files) == 0:
            raise RuntimeError(f"No .png files found under {self.root}")

    def __len__(self):
        return len(self.files)

    def _code_to_label(self, code: str) -> torch.Tensor:

        if ('_' in code):
            # Multiple angles, take only the code
            code = code.split('_')[0]

        if len(code) != len(self.schema.get_sliders()):
            raise ValueError(f"Code '{code}' length {len(code)} != 37")
        label_idxs = []
        for pos, ch in enumerate(code):
            ch = ch.upper()
            if ch not in self.char2idx[pos]:
                # Take the highest possible value (last index)
                max_idx = len(self.char2idx[pos]) - 1
                label_idxs.append(max_idx)
            else:
                label_idxs.append(self.char2idx[pos][ch])
        return torch.tensor(label_idxs, dtype=torch.long)


    def __getitem__(self, idx: int):
        path = self.files[idx]
        code = self.remove_periods(path.stem)
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self._code_to_label(code)
        return img, label, code
    
    def remove_periods(self, code_with_periods: str) -> str:
        """Remove periods from a code string."""
        return code_with_periods.replace('.', '')

class CodePredictor(nn.Module):
    def __init__(self, num_classes_per_digit: List[int]):
        super().__init__()
        # Try to load with weights; fallback to no weights if torch/torchvision versions differ
        try:
            backbone = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.DEFAULT)
        except Exception:
            backbone = torchvision.models.mobilenet_v3_small(pretrained=True)
        # Remove final classifier; keep features -> pooled embedding
        self.features = backbone.features
        # The classifier in mobilenet_v3_small: Sequential[Linear(576->1024), Hardswish, Dropout, Linear(1024->1000)]
        # We'll keep the first Linear + Hardswish to get a 1024-dim feature
        cls = backbone.classifier
        self.proj = nn.Sequential(cls[0], cls[1])  # Linear->Hardswish, outputs 1024
        in_features = 1024
        # Create one head per digit position
        self.heads = nn.ModuleList([nn.Linear(in_features, n) for n in num_classes_per_digit])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.features(x)
        # Global average pool
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.proj(x)
        outs = [head(x) for head in self.heads]
        return outs