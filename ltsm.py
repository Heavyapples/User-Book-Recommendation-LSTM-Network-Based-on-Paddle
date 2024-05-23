import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# 图书模型类
class Model(paddle.nn.Layer):
    def __init__(self, user_favorite_type, user_book_cat, user_gender_age_job, fc_sizes, hidden_size=64, num_layers=1):
        super(Model, self).__init__()

        self.user_favorite_type = user_favorite_type
        self.user_gender_age_job = user_gender_age_job
        self.user_book_cat = user_book_cat
        self.fc_sizes = fc_sizes
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        Dataset = DataProcessing()
        self.Dataset = Dataset
        self.trainset = self.Dataset.train_dataset
        self.valset = self.Dataset.valid_dataset
        self.train_loader = self.Dataset.load_data(dataset=self.trainset, mode='train')
        self.valid_loader = self.Dataset.load_data(dataset=self.valset, mode='valid')

        self.user_emb = nn.Embedding(num_embeddings=Dataset.max_user_id + 1, embedding_dim=32, sparse=False)
        self.user_lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, num_layers=num_layers)

        self.book_emb = nn.Embedding(num_embeddings=Dataset.max_book_id + 1, embedding_dim=32, sparse=False)
        self.book_lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, num_layers=num_layers)

        self.fc = nn.Linear(in_features=hidden_size*2, out_features=1)

    def get_user_feat(self, user_var):
        user_id, _, _, _, _ = user_var
        user_emb = self.user_emb(user_id)
        user_emb = user_emb.unsqueeze(0)
        _, (user_hidden, _) = self.user_lstm(user_emb)
        user_features = user_hidden[-1]
        return user_features

    def get_book_feat(self, book_var):
        book_id, _ = book_var
        book_emb = self.book_emb(book_id)
        book_emb = book_emb.unsqueeze(0)
        _, (book_hidden, _) = self.book_lstm(book_emb)
        book_features = book_hidden[-1]
        return book_features

    def forward(self, user_var, book_var):
        user_features = self.get_user_feat(user_var)
        book_features = self.get_book_feat(book_var)

        combined_features = paddle.concat((user_features, book_features), axis=1)
        res = self.fc(combined_features)
        
        return user_features, book_features, res
