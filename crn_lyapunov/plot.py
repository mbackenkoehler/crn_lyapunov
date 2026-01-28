import matplotlib.pyplot as plt
import seaborn as sns


def loss_traj(h_loss, h_dmax):
    fig, ax = plt.subplots()
    ax.plot(h_loss, label="loss")
    ax.set_yscale("log")
    ax.twinx().plot(h_dmax, label="max", c="r")
    plt.legend()


def drift_2d(x_max, y_max, num_points=300):
    x_range = torch.linspace(0, x_max, num_points)
    y_range = torch.linspace(0, y_max, num_points)

    x_mesh, y_mesh = torch.meshgrid(x_range, y_range, indexing="xy")

    x_grid = torch.stack([x_mesh.flatten(), y_mesh.flatten()], dim=1).to(device).float()

    with torch.no_grad():
        grid_drift = get_drift(model, net, x_grid).cpu().numpy()
        grid_drift_ref = get_drift(ref, net, x_grid).cpu().numpy()

    drift_heatmap_data = grid_drift.reshape(num_points, num_points) / grid_drift.max()
    drift_heatmap_data_ref = (
        grid_drift_ref.reshape(num_points, num_points) / grid_drift_ref.max()
    )

    # Create the heatmap plot
    plt.figure(figsize=(10, 8))
    plt.imshow(
        drift_heatmap_data,
        origin="lower",
        extent=[
            x_range.min(),
            x_range.max(),
            y_range.min(),
            y_range.max(),
        ],
        cmap="viridis",
        aspect="auto",
    )

    plt.colorbar(label="Drift Value")

    points_lv = adv_parbd.population.detach().cpu().numpy()
    plt.scatter(
        points_lv[:, 0],
        points_lv[:, 1],
        c="red",
        s=5,
        alpha=0.5,
        label="Adversarial Points",
    )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
