import argparse
import json
import os
import torch
import sys
sys.path.append("./")
from torch.utils.data import DataLoader
from tqdm import tqdm
from DNN_spec.DNNnet import DNNnet, Dnn_net_Loss
from DNN_spec.spec2load import spec2load


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded checkpoint '{}' (iteration {})" .format(
          checkpoint_path, iteration))
    return model, optimizer, iteration

def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
          iteration, filepath))
    model_for_saving = DNNnet(**DNN_net_config).cuda()
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


                
def DNN_train(output_directory, epochs, learning_rate,\
    iters_per_checkpoint, batch_size, seed, checkpoint_path):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # MMSELoss
    criterion = Dnn_net_Loss()

    model = DNNnet(**DNN_net_config).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Load checkpoint if one exists
    iteration = 0
    if checkpoint_path != "":
        model, optimizer, iteration = load_checkpoint(checkpoint_path, model,
                                                      optimizer)
        model.cuda()
        iteration += 1  # next iteration is iteration + 1

    trainSet = spec2load(**DNN_data_config)
    trainSet.load_buffer()
    train_loader = DataLoader(trainSet, num_workers=1, shuffle=True,
                            batch_size=batch_size,
                            pin_memory=False,
                            drop_last=True)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory)
    model.train(mode=True)
    epoch_offset = max(0, int(iteration / len(train_loader)))
    
    for epoch in range(epoch_offset, epochs):
        epoch_ave_loss = 0
        for i, batch in tqdm(enumerate(train_loader)):
            model.zero_grad()

            feed_in, targ_in = batch
            feed_in = torch.autograd.Variable(feed_in.cuda())
            targ_in = torch.autograd.Variable(targ_in.cuda())
            outputs = model(feed_in)

            loss = criterion(outputs, targ_in)

            reduced_loss = loss.item()

            loss.backward()

            optimizer.step()

            epoch_ave_loss += reduced_loss

            if (iteration % iters_per_checkpoint == 0):
                print("{}:\t{:.9f}".format(iteration, reduced_loss))
            
            iteration += 1

        checkpoint_path = "{}/DNN_net_{}".format(
            output_directory, epoch)
        save_checkpoint(model, optimizer, learning_rate, iteration,
                        checkpoint_path)
        epoch_ave_loss = epoch_ave_loss / i
        print("Epoch: {}, the average epoch loss: {}".format(epoch, epoch_ave_loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='DNN_spec/config.json',
                        help='JSON file for configuration')

    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    DNN_train_config = config["DNN_train_config"]
    global DNN_data_config
    DNN_data_config = config["DNN_data_config"]
    global DNN_net_config
    DNN_net_config = config["DNN_net_config"]


    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    DNN_train(**DNN_train_config)
