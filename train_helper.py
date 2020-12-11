from models import Model
from dataset import Dataset2
from data_utils import Util

import torch
import ntpath



def train(save_folder, restore_from, loggers):
  try:
    model = Model().to(Util.device)
    if restore_from:
      loggers[1](f"Restore model from {restore_from} ...")
      start_epoch = int(ntpath.basename(restore_from).split("-")[1].split(".pt")[0])
      model.load_state_dict(torch.load(restore_from))
      model.eval()
    else:
      start_epoch=-1
      loggers[1]("Create model from scratch ...")

    loggers[1]("load dataset ...")
    dataset = Dataset2()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=Util.learning_rate)

    for epoch in range(start_epoch+1, Util.epochs):
      loggers[1](f"**** EPOCH {epoch} ****")
      L = 0
      for i, batch in enumerate(dataset):
        batch["points"]  = torch.tensor(batch["points"]).to(Util.device)
        batch["camera_poses"]  = torch.tensor(batch["camera_poses"]).to(Util.device)
        batch["segmentation"]  = torch.tensor(batch["segmentation"]).to(Util.device)

        output = model(batch)
        losses = model.compute_losses(batch, output)
        total_loss = sum(losses.values())
        L+=total_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # replace by logger_0
        loggers[1](f"Epoch:{epoch}, Batch:{i} - Tot.L:{total_loss.item():.4f},"+\
                    f" Recons. L:{losses['reconstruction_loss'].item():.4f}, "+\
                    f"Seg. L:{losses['segmentation_loss'].item():.4f}, "+\
                    f"kl_1:{losses['kl_1'].item():.4f}, "\
                    f"kl_t:{losses['kl_t'].item():.4f}")

      loggers[1](f"Epoch:{epoch}, Total loss : {L.item()/(i+1)}")

      ## Save checkpoint
      torch.save(model.state_dict(), save_folder+f"/model-{epoch}.pt")
  except KeyboardInterrupt:
    loggers[1]("Training Intereputed ...")
