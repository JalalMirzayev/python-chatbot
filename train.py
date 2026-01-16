import torch
from utils import load_intents
from utils import get_words
from utils import prepare_data
from chatbot_dataset import ChatbotDataset
from torch.utils.data import DataLoader
from model import Model


if __name__ == '__main__':
    data = load_intents(path='intents.json')
    words_unique, tags_unique = get_words(data)
    X_train, y_train = prepare_data(words=words_unique, tags=tags_unique, data=data)
    
    learning_rate = 0.001
    epochs = 1000
    batch_size = 1
    input_size = len(words_unique)
    hidden_size = 8
    output_size = len(tags_unique)
    dataset = ChatbotDataset(X_train, y_train)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(input_size, hidden_size, output_size)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device)

            outputs = model(words)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'epoch {epoch + 1}/{epochs}, loss={loss.item():.5f}')


print(f'final loss, loss={loss.item():.5f}')

