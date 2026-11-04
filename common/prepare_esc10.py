"""
 Usages: Membuat dataset ESC-10 (20kHz) dari ESC-50 yang SUDAH disiapkan oleh
         common/prepare_dataset.py. Format output sama persis dengan
         datasets/esc50/wav20.npz, jadi trainer / train_generator bawaan repo
         bisa langsung memakainya (--dataset esc10).

 Input  : datasets/esc50/ESC-50-master/meta/esc50.csv   (kolom esc10 == True)
          datasets/esc50/wav20/*.wav                     (hasil ffmpeg 20kHz)
 Output : datasets/esc10/wav20.npz
          datasets/esc10/classes.txt                     (peta label baru -> nama kelas)

 Jalankan dari root repo:  python common/prepare_esc10.py
"""
import os
import csv
import numpy as np
import wavio


def main():
    root = os.getcwd()
    esc50_dir = os.path.join(root, 'datasets', 'esc50')
    esc10_dir = os.path.join(root, 'datasets', 'esc10')
    wav_dir = os.path.join(esc50_dir, 'wav20')
    meta_path = os.path.join(esc50_dir, 'ESC-50-master', 'meta', 'esc50.csv')

    for p in (wav_dir, meta_path):
        if not os.path.exists(p):
            raise SystemExit('Tidak ditemukan: {}\nJalankan dulu: python common/prepare_dataset.py'.format(p))
    os.makedirs(esc10_dir, exist_ok=True)

    # 1. Filter metadata: hanya klip ESC-10
    with open(meta_path, newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['esc10'].strip().lower() == 'true']

    # 2. Remap label ke 0..9 (target asli ESC-50 tidak kontigu, mis. 0,1,10,11,12,...)
    old_ids = sorted({int(r['target']) for r in rows})
    remap = {old: new for new, old in enumerate(old_ids)}
    names = {int(r['target']): r['category'] for r in rows}

    # 3. Sanity check sesuai dataset ESC-10: 400 klip, 10 kelas x 40, 5 fold x 80
    assert len(rows) == 400, 'ESC-10 harus 400 klip, dapat {}'.format(len(rows))
    assert len(old_ids) == 10
    for fold in range(1, 6):
        assert sum(int(r['fold']) == fold for r in rows) == 80, 'fold {} tidak 80 klip'.format(fold)

    # 4. Bangun dataset dengan format yang sama seperti common/prepare_dataset.py
    dataset = {}
    for fold in range(1, 6):
        sounds, labels = [], []
        for r in sorted((r for r in rows if int(r['fold']) == fold), key=lambda r: r['filename']):
            sound = wavio.read(os.path.join(wav_dir, r['filename'])).data.T[0]
            nz = sound.nonzero()[0]
            sound = sound[nz.min(): nz.max() + 1]  # buang bagian senyap (sama seperti repo)
            sounds.append(sound)
            labels.append(remap[int(r['target'])])
        dataset['fold{}'.format(fold)] = {'sounds': sounds, 'labels': labels}
        print('fold{}: {} klip, label unik {}'.format(fold, len(sounds), sorted(set(labels))))

    out_path = os.path.join(esc10_dir, 'wav20.npz')
    np.savez(out_path, **dataset)
    with open(os.path.join(esc10_dir, 'classes.txt'), 'w') as f:
        for old in old_ids:
            f.write('{}\t{}\t{}\n'.format(remap[old], old, names[old]))
    print('Tersimpan:', out_path)


if __name__ == '__main__':
    main()