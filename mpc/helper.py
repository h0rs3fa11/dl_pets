import crypten
import crypten.mpc as mpc
import crypten.communicator as comm
import torch
import numpy as np
import os

crypten.init()
torch.set_num_threads(1)

parent_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..'))

@mpc.run_multiprocess(world_size=2)
def split_input_data(input_type='train', save_dir_template="party{rank}/{input_type}"):
    rank = comm.get_rank()  # Party's rank (0 or 1)

    ALICE = 0  
    BOB = 1

    if input_type == 'train':
        filename = 'train_features.npz'
    else:
        filename = 'val_features.npz'

    npz_file_path = os.path.join(parent_dir, 'data', 'extracted_features', filename)

    current_party_save_dir = save_dir_template.format(rank=rank, input=input_type)
    os.makedirs(current_party_save_dir, exist_ok=True)

    crypten.print(f"Party {rank}: Save directory set to {current_party_save_dir}")

    try:
        crypten.print(f"Party {rank}: Loading data from {npz_file_path}...")
        data = np.load(npz_file_path)
        X_full_np = data['X']
        y_full_np = data['y']
        crypten.print(f"Party {rank}: Data loaded. X shape: {X_full_np.shape}, y shape: {y_full_np.shape}")
    except FileNotFoundError:
        crypten.print(f"ERROR - Party {rank}: Could not find or load {npz_file_path}.")
        comm.get().barrier() # Ensure all parties see error and exit somewhat gracefully
        return
    except Exception as e:
        crypten.print(f"ERROR - Party {rank}: Failed to load data from {npz_file_path}. Error: {e}")
        comm.get().barrier()
        return

    num_total_features = X_full_np.shape[1]
    split_idx = num_total_features // 2

    X1_np = X_full_np[:, :split_idx]
    X2_np = X_full_np[:, split_idx:]

    y_np = y_full_np

    crypten.print(f"Party {rank}: X_full_np features: {num_total_features}, split_idx: {split_idx}")
    crypten.print(f"Party {rank}: X1_np shape: {X1_np.shape}, X2_np shape: {X2_np.shape}, y_np shape: {y_np.shape}")

    if rank == ALICE:
        X_party_pt = torch.from_numpy(X1_np).float()
        y_party_pt = torch.from_numpy(y_np).float()
    elif rank == BOB:
        X_party_pt = torch.from_numpy(X2_np).float()
    else:
        # Should not happen with world_size=2
        crypten.print(f"Party {rank}: Error - Unexpected rank.")
        comm.get().barrier()
        return

    if rank == ALICE:
        crypten.print(f"Party {ALICE}: Encrypting X1 (shape: {X_party_pt.shape}) and y (shape: {y_party_pt.shape})...")
        x1_enc = crypten.cryptensor(X_party_pt, src=ALICE)
        y_enc = crypten.cryptensor(y_party_pt, src=ALICE)

        # Save Alice's encrypted shares
        x1_save_path = os.path.join(current_party_save_dir, "X1.pt")
        y_save_path = os.path.join(current_party_save_dir, "y.pt")
        crypten.save_from_party(x1_enc, x1_save_path, src=ALICE)
        crypten.save_from_party(y_enc, y_save_path, src=ALICE)
        crypten.print(f"Party {ALICE}: Saved its encrypted X1 to {x1_save_path}")
        crypten.print(f"Party {ALICE}: Saved its encrypted y to {y_save_path}")

    elif rank == BOB:
        crypten.print(f"Party {BOB}: Encrypting X2 (shape: {X_party_pt.shape})...")
        x2_enc = crypten.cryptensor(X_party_pt, src=BOB)

        # Save Bob's encrypted share
        x2_save_path = os.path.join(current_party_save_dir, "X2.pt")
        crypten.save_from_party(x2_enc, x2_save_path, src=BOB)
        crypten.print(f"Party {BOB}: Saved its encrypted X2 to {x2_save_path}")

    comm.get().barrier()
    print(f"Party {rank}: Processing finished successfully.")

split_input_data('train')