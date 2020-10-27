from __future__ import print_function, division
from keras.datasets import mnist
from instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import to_categorical
import scipy

import matplotlib.pyplot as plt

import numpy as np

class SASGAN():
    def __init__(self):
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.mask_height = 10
        self.mask_width = 10
        self.num_classes = 10
        self.df = 32

        # filters options
        self.fs_1 = 32
        self.fs_2 = 64
        self.fs_3 = 128
        self.fs_4 = 256

        # kernel_size options
        self.ks_1 = 3
        self.ks_2 = 4
        self.ks_3 = 5

        # strides options
        self.strides_1 = 1
        self.strides_2 = 2

        # drop-out options
        self.dout_1 = 0
        self.dout_2 = 0.1
        self.dout_3 = 0.5

        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.Discriminator()
        self.discriminator.compile(loss=['mse', 'categorical_crossentropy'],
            loss_weights=[0.5, 0.5],
            optimizer=optimizer,
            metrics=['accuracy'])

        self.generator = self.Generator()

        masked_img = Input(shape=self.img_shape)
        gen_img = self.generator(masked_img)

        self.discriminator.trainable = False

        valid, _ = self.discriminator(gen_img)

        self.combined = Model(masked_img , valid)
        self.combined.compile(loss=['mse'],
            optimizer=optimizer)


    def Generator(self):

        def block_conv(layer_input, filters, kernel_size, strides, bn=True):
            """Layers used during downsampling"""
            cell = Conv2D(filters, kernel_size, strides, padding='same')(layer_input)
            cell = LeakyReLU(alpha=0.2)(cell)
            if bn:
                cell = BatchNormalization(momentum=0.8)(cell)
            return cell

        def block_deconv(layer_input, skip_input, filters, kernel_size, strides, dropout_rate=0):
            """Layers used during upsampling"""
            cell = UpSampling2D(size=2)(layer_input)
            cell = Conv2D(filters, kernel_size, strides, padding='same', activation='relu')(cell)
            if dropout_rate:
                cell = Dropout(dropout_rate)(cell)
            cell = BatchNormalization(momentum=0.8)(cell)
            cell = Concatenate()([cell, skip_input])
            return cell

        img = Input(shape=self.img_shape)

        d1 = block_conv(img, filters=self.fs_1, kernel_size=self.ks_2, strides=self.strides_2, bn=False)
        d2 = block_conv(d1, filters=self.fs_2, kernel_size=self.ks_2, strides=self.strides_2)
        d3 = block_conv(d2, filters=self.fs_3, kernel_size=self.ks_2, strides=self.strides_2)
        d4 = block_conv(d3, filters=self.fs_4, kernel_size=self.ks_2, strides=self.strides_2)


        u1 = block_deconv(d4, d3, self.fs_4, kernel_size=self.ks_1, strides=self.strides_1)
        u2 = block_deconv(u1, d2, self.fs_2, kernel_size=self.ks_1, strides=self.strides_1)
        u3 = block_deconv(u2, d1, self.fs_1, kernel_size=self.ks_1, strides=self.strides_1)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(img, output_img)

    def Discriminator(self):

        img = Input(shape=self.img_shape)

        model = Sequential()
        model.add(Conv2D(64, kernel_size=4, strides=2, padding='same', input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.8))
        model.add(Conv2D(128, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(InstanceNormalization())
        model.add(Conv2D(256, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(InstanceNormalization())

        model.summary()

        img = Input(shape=self.img_shape)
        features = model(img)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(features)

        label = Flatten()(features)
        label = Dense(self.num_classes+1, activation="softmax")(label)

        return Model(img, [validity, label])

    def mask_randomly(self, imgs):
        y1 = np.random.randint(0, self.img_rows - self.mask_height, imgs.shape[0])
        y2 = y1 + self.mask_height
        x1 = np.random.randint(0, self.img_rows - self.mask_width, imgs.shape[0])
        x2 = x1 + self.mask_width

        masked_imgs = np.empty_like(imgs)
        for i, img in enumerate(imgs):
            masked_img = img.copy()
            _y1, _y2, _x1, _x2 = y1[i], y2[i], x1[i], x2[i],
            masked_img[_y1:_y2, _x1:_x2, :] = 0
            masked_imgs[i] = masked_img

        return masked_imgs


    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, y_train), (x_test,y_test) = mnist.load_data()

        # Rescale MNIST to 32x32
        X_train = np.array([scipy.misc.imresize(x, [self.img_rows, self.img_cols]) for x in X_train])

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        x_test = (x_test.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        x_test = np.expand_dims(x_test, axis=3)
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 4, 4, 1))
        fake = np.zeros((batch_size, 4, 4, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Sample half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            idt = np.random.randint(0, x_test.shape[0], 32)
            test_imgs = x_test[idt]
            labels = y_train[idx]
            test_labels = y_test[idt]

            masked_imgs = self.mask_randomly(imgs)

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(masked_imgs)

            # One-hot encoding of labels
            labels = to_categorical(labels, num_classes=self.num_classes+1)
            test_labels = to_categorical(test_labels, num_classes=self.num_classes+1)
            fake_labels = to_categorical(np.full((batch_size, 1), self.num_classes), num_classes=self.num_classes+1)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, labels])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, fake_labels])
            # d_loss =  0.2 * np.add(d_loss_real, 0) + 0.8 * np.add(d_loss_fake, 0)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train the generator
            g_loss = self.combined.train_on_batch(masked_imgs, valid)

            # Plot the progress
            print ("%d [D loss: %f, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[4], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                # Select a random half batch of images
                idx = np.random.randint(0, X_train.shape[0], 6)
                imgs = X_train[idx]
                self.sample_images(epoch, imgs)
                self.save_model()

            # ---------------------
            #  Test process
            # ---------------------
            # test every 100 epochs
            if epoch % 100 == 0:
                Loss_test = self.discriminator.train_on_batch(test_imgs,[valid,test_labels])
                print ("The epoch %d test result is: [D loss: %f, op_acc: %.2f%%] " % (epoch, Loss_test[0], 100*d_loss[4]))

    def sample_images(self, epoch, imgs):
        r, c = 3, 6

        masked_imgs = self.mask_randomly(imgs)
        gen_imgs = self.generator.predict(masked_imgs)

        imgs = (imgs + 1.0) * 0.5
        masked_imgs = (masked_imgs + 1.0) * 0.5
        gen_imgs = (gen_imgs + 1.0) * 0.5

        gen_imgs = np.where(gen_imgs < 0, 0, gen_imgs)

        fig, axs = plt.subplots(r, c)
        for i in range(c):
            axs[0,i].imshow(imgs[i, :, :, 0], cmap='gray')
            axs[0,i].axis('off')
            axs[1,i].imshow(masked_imgs[i, :, :, 0], cmap='gray')
            axs[1,i].axis('off')
            axs[2,i].imshow(gen_imgs[i, :, :, 0], cmap='gray')
            axs[2,i].axis('off')
        fig.savefig("images/%d.png" % epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "sasgan_generator")
        save(self.discriminator, "sasgan_discriminator")


if __name__ == '__main__':
    sasgan = SASGAN()
    sasgan.train(epochs=20000, batch_size=32, sample_interval=200)
