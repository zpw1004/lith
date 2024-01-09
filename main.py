import torch
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from util import get_confusion_matrix, parse_arguments, save_matrix, write_file
from load_data import get_daqing, get_xinjiang, get_blind
from model import MultiScaleNetwork, multi_FocalLoss
from train import train_model, evaluate
def main():
    args = parse_arguments()
    if args.dataset == "daqing":
        x_train, y_train, data_trainloader, x_test, y_test, data_test_loader = get_daqing(args.data_path1)
        model_params = (args.num_classes, args.features1)
        focal_loss_weights = args.weights1
        save_path = './datasave/daqing/'
    elif args.dataset == "xinjiang":
        x_train, y_train, data_trainloader, x_test, y_test, data_test_loader = get_xinjiang(args.data_path2)
        model_params = (args.num_classes, args.features2)
        focal_loss_weights = args.weights2
        save_path = './datasave/xinjiang/'
    elif args.dataset == "blind":
        x_train, y_train, data_trainloader, x_test, y_test, data_test_loader = get_blind(args.data_path1)
        model_params = (args.num_classes, args.features1,0.6)
        focal_loss_weights = args.weights3
        save_path = './datasave/blind/'
    else:
        raise ValueError("Invalid dataset name")
    times = 1
    seed = args.seed
    train_accs = []
    test_accs = []
    for j in range(times):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        model = MultiScaleNetwork(*model_params)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        criterion = multi_FocalLoss(5, focal_loss_weights)
        best_accuracy = 0
        best_model_path = save_path + f'{args.dataset}_best.pth'
        device = torch.device("cuda:0" if args.cuda else "cpu")
        model.to(device)
        for epoch in range(1,args.epochs+1):
            train_model(model, criterion, optimizer, data_trainloader, device=device)
            train_accuracy, _ = evaluate(model, x_train, y_train, device=device)
            test_accuracy, predicted_test = evaluate(model, x_test, y_test, device=device)
            train_accs.append(train_accuracy)
            test_accs.append(test_accuracy)
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(model.state_dict(), best_model_path)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, LR: {scheduler.get_last_lr()[0]}, Train Acc: {train_accuracy}, Test Acc: {test_accuracy}")
            scheduler.step()
        model.load_state_dict(torch.load(best_model_path))
        accuracy, predicted = evaluate(model, x_test, y_test, device=device)
        path = save_path + 'y_pre/multi_input.txt'
        write_file(path,predicted)
        precision = precision_score(y_test.cpu(), predicted.cpu(), average='macro')
        recall = recall_score(y_test.cpu(), predicted.cpu(), average='macro')
        f1 = f1_score(y_test.cpu(), predicted.cpu(), average='macro')
        conf_matrix = get_confusion_matrix(y_test.cpu(), predicted.cpu())
        print(f"Run {j + 1}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
if __name__ == '__main__':
    main()