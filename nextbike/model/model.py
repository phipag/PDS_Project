from sklearn.linear_model import LinearRegression

from nextbike import io


def train():
    lin = LinearRegression()
    print('Linear model created')
    print('Training...')

    io.save_model(lin)
