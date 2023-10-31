### Loss functions


def mse_loss(output, target):
    return ((output - target) ** 2).mean()


def weighted_binary_cross_entropy(output, target, weight=None):
    if weight is None:
        return F.binary_cross_entropy(output, target)
    else:
        return F.binary_cross_entropy(output, target, weight=weight)


def focal_loss(output, target, alpha=1, gamma=2):
    ce_loss = F.binary_cross_entropy(output, target, reduction="none")
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return torch.mean(focal_loss)


def dice_loss(output, target, smooth=1e-5):
    intersection = (output * target).sum()
    union = output.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice


# Tversky loss is an extension of DICE loss
def tversky_loss(output, target, alpha=0.5, beta=0.5, smooth=1e-5):
    """Parameters:
    alpha, beta: Control the magnitude of penalties for false positives and false negatives.
    alpha is used for false positives, and beta is used for false negatives.
    The default value of 0.5 for both makes it equivalent to the DICE loss."""
    intersection = (output * target).sum()
    fp = (output * (1 - target)).sum()
    fn = ((1 - output) * target).sum()
    tversky_index = (intersection + smooth) / (
        intersection + alpha * fp + beta * fn + smooth
    )
    return 1 - tversky_index


# Matthew's Correlation Coefficient
def mcc_loss(output, target, smooth=1e-5):
    tp = (output * target).sum()
    tn = ((1 - output) * (1 - target)).sum()
    fp = (output * (1 - target)).sum()
    fn = ((1 - output) * target).sum()
    numerator = (tp * tn) - (fp * fn)
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + smooth)
    mcc = numerator / (denominator + smooth)
    return 1 - mcc


# Asymmetric imposes larger penalty on false negatives than false positives
def asymmetric_loss(output, target, gamma_neg=4, gamma_pos=1):
    """gamma_neg, gamma_pos: Control the focus on false negatives and false positives.
    Higher values of gamma_neg or gamma_pos will focus more on minimizing the
    corresponding type of error. For example, if false negatives are more costly than
    false positives, you might set gamma_neg higher than gamma_pos."""
    loss = -target * torch.log(output + 1e-5) * torch.pow((1 - output), gamma_pos) - (
        1 - target
    ) * torch.log(1 - output + 1e-5) * torch.pow(output, gamma_neg)
    return loss.mean()


# Balance classes by re-weighting
def class_balanced_loss(output, target, num_classes, beta=0.9999):
    """beta: Controls the re-weighting. A smaller beta will produce weights that are
    closer to the class frequency. It's a hyperparameter between 0 and 1 that represents
    the class frequency in your dataset. A commonly used value is 0.9999."""

    effective_num = 1.0 - torch.pow(beta, num_classes)
    weights = (1.0 - beta) / torch.tensor(effective_num)
    weights = weights / torch.sum(weights) * num_classes

    focal_loss = F.binary_cross_entropy(output, target, reduction="none")
    weighted_loss = torch.mul(weights, focal_loss)
    loss = torch.sum(weighted_loss) / torch.sum(weights)
    return loss


### Train data and plot results


# Define your train_dataset and validation_dataset here
train_dataset = ImageToImageDataset(
    train_input_dir,
    train_output_dir,
    input_transforms=dataTransformer,
    output_transform=grayscaleTransform,
)
validation_dataset = ImageToImageDataset(
    val_input_dir,
    val_output_dir,
    input_transforms=dataTransformer,
    output_transform=grayscaleTransform,
)

# Define model
model = RNASecondaryStructureCAE().cuda()

loss_functions = {
    "Weighted BCE": weighted_binary_cross_entropy,
    "Focal Loss": focal_loss,
    "MSE": mse_loss,
    "DICE Loss": dice_loss,
    "Tversky Loss": tversky_loss,
    "Matthew's Correlation Coefficient": mcc_loss,
    "Asymmetric Loss": asymmetric_loss,
    "Class Balanced Loss": class_balanced_loss,
}


training_losses = {name: [] for name in loss_functions.keys()}
validation_losses = {name: [] for name in loss_functions.keys()}


def plot_loss_curves(training_losses, validation_losses):
    plt.figure(figsize=(10, 6))

    for loss_name in training_losses.keys():
        plt.plot(
            training_losses[loss_name], label=f"Training {loss_name}", linestyle="-"
        )

    for loss_name in validation_losses.keys():
        plt.plot(
            validation_losses[loss_name],
            label=f"Validation {loss_name}",
            linestyle="--",
        )

    plt.title("Training and Validation Losses for Different Loss Functions")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend(loc="upper right")
    plt.show()


for loss_name, loss_fn in loss_functions.items():
    print(f"Training with {loss_name}...")
    train_loss, train_F1, valid_loss, valid_F1, plot_time = fit_model(
        model,
        train_dataset,
        validation_dataset,
        loss_func=loss_fn,
        optimizer=adam_optimizer,
        lr=0.01,
        bs=1,
        epochs=1,
    )
    training_losses[loss_name].extend(train_loss)
    validation_losses[loss_name].extend(valid_loss)

plot_loss_curves(training_losses, validation_losses)
