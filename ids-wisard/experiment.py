import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


from datasetutils import load_dataset, split_dataset
from binarization import preprocess_features, apply_preprocess_to_test, thermometer_encode
from wisardmodel import WiSARD
from baselines import train_random_forest, train_svm, train_knn

''' Seria interessante ativar essa função caso queira gerar apenas os csvs da wisard
def save_wisard_prediction_csvs(X_test_df, y_test_enc, y_pred_enc, label_encoder,
                                output_dir="results/predictions"):

    os.makedirs(output_dir, exist_ok=True)

    y_true = label_encoder.inverse_transform(y_test_enc)
    y_pred = label_encoder.inverse_transform(y_pred_enc)

    df = X_test_df.copy()
    df["TrueLabel"] = y_true
    df["PredLabel"] = y_pred

    for label in sorted(df["PredLabel"].unique()):
        df_label = df[df["PredLabel"] == label]
        out_path = os.path.join(output_dir, f"wisardpred{label}.csv")
        df_label.to_csv(out_path, index=False)
        print(f"CSV salvo com amostras que a WiSARD classificou como {label}: {out_path}")
'''

def save_prediction_distribution_plot(y_pred_enc, label_encoder, model_name,
                                      output_dir="results/plots"):

    os.makedirs(output_dir, exist_ok=True)

    labels = label_encoder.inverse_transform(y_pred_enc)
    counts = pd.Series(labels).value_counts().sort_index()

    plt.figure(figsize=(8, 4))
    plt.plot(counts.index, counts.values, marker='o', linestyle='-', linewidth=2)

    plt.title(f"Distribuição de classes{model_name}")
    plt.xlabel("Classe")
    plt.ylabel("Número de amostras")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{model_name.lower().replace(' ', '')}preddistribution.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Gráfico de distribuição ({model_name}) salvo em: {out_path}")



def save_wisardheatmap(results_wisard, output_dir="results/plots"):
    os.makedirs(output_dir, exist_ok=True)

    bits = sorted({r["n_bits"] for r in results_wisard})
    tuples = sorted({r["tuple_size"] for r in results_wisard})

    mat = np.full((len(bits), len(tuples)), np.nan)

    for r in results_wisard:
        i = bits.index(r["n_bits"])
        j = tuples.index(r["tuple_size"])
        mat[i, j] = r["accuracy"] * 100.0

    plt.figure()
    im = plt.imshow(mat, aspect="auto", origin="lower")
    plt.colorbar(im, label="Acurácia (%)")
    plt.xticks(range(len(tuples)), tuples)
    plt.yticks(range(len(bits)), bits)
    plt.xlabel("tuple_size")
    plt.ylabel("N_BITS")
    plt.title("Acurácia da WiSARD por N_BITS e tuple_size")

    for i in range(len(bits)):
        for j in range(len(tuples)):
            if not np.isnan(mat[i, j]):
                plt.text(j, i, f"{mat[i, j]:.1f}",
                         ha="center", va="center", fontsize=6, color="white")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "wisardheatmap.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Heatmap da WiSARD salvo em: {out_path}")


def save_wisardaccuracycurves(results_wisard, output_dir="results/plots"):
    os.makedirs(output_dir, exist_ok=True)

    bits = sorted({r["n_bits"] for r in results_wisard})
    tuples = sorted({r["tuple_size"] for r in results_wisard})

    plt.figure()
    for b in bits:
        xs, ys = [], []
        for t in tuples:
            match = next((r for r in results_wisard
                          if r["n_bits"] == b and r["tuple_size"] == t), None)
            if match is not None:
                xs.append(t)
                ys.append(match["accuracy"] * 100.0)

        if xs:
            plt.plot(xs, ys, marker="o", label=f"N_BITS={b}")

    plt.xlabel("tuple_size")
    plt.ylabel("Acurácia (%)")
    plt.title("Curvas de acurácia da WiSARD por N_BITS")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(output_dir, "wisardaccuracycurves.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Curvas de acurácia da WiSARD salvas em: {out_path}")

from sklearn.metrics import confusion_matrix
import seaborn as sns

def save_confusion_matrix(y_true_enc, y_pred_enc, label_encoder,
                          model_name, output_dir="results/confusionmatrix"):

    os.makedirs(output_dir, exist_ok=True)

    labels = label_encoder.classes_

    cm = confusion_matrix(y_true_enc, y_pred_enc, labels=range(len(labels)))

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)

    plt.title(f"Matriz de Confusão - {model_name}")
    plt.xlabel("Predição")
    plt.ylabel("Verdadeiro")

    out_path = os.path.join(output_dir, f"{model_name.lower().replace(' ', '')}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Matriz de confusão ({model_name}) salva em: {out_path}")

def save_combined_confusion_matrices(
    y_true,
    y_pred_wisard,
    y_pred_rf,
    y_pred_svm,
    y_pred_knn,
    label_encoder,
    output_path="results/confusionmatrix/all_models_confusion.png",
):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    labels = label_encoder.classes_

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    modelos = [
        ("WiSARD",      y_pred_wisard, axes[0, 0]),
        ("RandomForest", y_pred_rf,    axes[0, 1]),
        ("SVM",         y_pred_svm,   axes[1, 0]),
        ("KNN",         y_pred_knn,   axes[1, 1]),
    ]

    for nome, y_pred, ax in modelos:
        cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(nome)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Matriz de confusão comparativa salva em: {output_path}")

def save_model_prediction_csvs(X_test_df, y_test_enc, y_pred_enc, label_encoder,
                               model_name, output_dir="results/model_predictions"):

    os.makedirs(output_dir, exist_ok=True)

    y_true = label_encoder.inverse_transform(y_test_enc)
    y_pred = label_encoder.inverse_transform(y_pred_enc)

    df = X_test_df.copy()
    df["TrueLabel"] = y_true
    df["PredLabel"] = y_pred

    safe_name = model_name.lower().replace(" ", "")
    out_path = os.path.join(output_dir, f"{safe_name}predictions.csv")

    df.to_csv(out_path, index=False)

    print(f"CSV de predições ({model_name}) salvo em: {out_path}")

def save_class_split_predictions(X_test_df, y_test_enc, y_pred_enc, label_encoder,
                                 model_name, output_dir="results/predictions"):

    os.makedirs(output_dir, exist_ok=True)

    y_true = label_encoder.inverse_transform(y_test_enc)
    y_pred = label_encoder.inverse_transform(y_pred_enc)

    df = X_test_df.copy()
    df["TrueLabel"] = y_true
    df["PredLabel"] = y_pred

    for pred_label in sorted(df["PredLabel"].unique()):
        df_label = df[df["PredLabel"] == pred_label]

        fname = f"{model_name.lower().replace(' ', '')}{pred_label.lower()}.csv"
        out_path = os.path.join(output_dir, fname)

        df_label.to_csv(out_path, index=False)
        print(f"CSV {model_name} = {pred_label}: {out_path}")

def main():
    print("Carregamento do dataset IoT-SDN IDS")
    X, y = load_dataset()
    print("Shape original:", X.shape)
    print("Total amostras:", len(y))
    print("Distribuição de classes (label):")
    print(y.value_counts())

    print("\nSplit treino/teste")
    X_train_df, X_test_df, y_train, y_test, label_encoder = split_dataset(X, y, test_size=0.3)
    print("Treino:", X_train_df.shape, "Teste:", X_test_df.shape)
    print("\nRemoção de features com variância muito baixa")

    selector = VarianceThreshold(threshold=0.0) 
    X_train_red = selector.fit_transform(X_train_df)
    X_test_red = selector.transform(X_test_df)

    print("Shape antes:", X_train_df.shape)
    print("Shape depois:", X_train_red.shape)

    print("Classes codificadas:", list(label_encoder.classes_))

    print("\nPré-processamento normalização e one-hot se for necessário")
    X_train_scaled, scaler, feature_order = preprocess_features(pd.DataFrame(X_train_red))
    X_test_scaled = apply_preprocess_to_test(pd.DataFrame(X_test_red), scaler, feature_order)
    print("Shape pós-pre-processamento:", X_train_scaled.shape)

    np.save("X_train_scaled.npy", X_train_scaled)
    np.save("X_test_scaled.npy", X_test_scaled)

    bits_options = [2, 3, 4, 5, 6, 8, 16]
    tuple_sizes = [8, 16, 24, 32, 40, 48]

    best_acc = 0.0
    best_conf = None
    best_y_pred_enc = None
    results_wisard = []

    print("\nMelhores hiperparâmetros obtidos na WiSARD")

    for N_BITS in bits_options:
        print(f"\n>> Testando N_BITS = {N_BITS}")
        X_train_bin = thermometer_encode(X_train_scaled, n_bits=N_BITS)
        X_test_bin = thermometer_encode(X_test_scaled, n_bits=N_BITS)

        rng = np.random.RandomState(42)
        perm = rng.permutation(X_train_bin.shape[1])
        X_train_bin = X_train_bin[:, perm]
        X_test_bin = X_test_bin[:, perm]

        input_size = X_train_bin.shape[1]
        n_classes = len(label_encoder.classes_)

        for tuple_size in tuple_sizes:
            if input_size < tuple_size:
                continue

            wisard = WiSARD(input_size=input_size,
                            tuple_size=tuple_size,
                            n_classes=n_classes)

            wisard.fit(X_train_bin, y_train)
            y_pred_wisard = wisard.predict(X_test_bin)
            acc = (y_pred_wisard == y_test).mean()

            print(f"  tuple_size={tuple_size} -> Acurácia WiSARD = {acc * 100:.2f}%")

            results_wisard.append({
                "n_bits": N_BITS,
                "tuple_size": tuple_size,
                "accuracy": acc
            })

            if acc > best_acc:
                best_acc = acc
                best_conf = (N_BITS, tuple_size)
                best_y_pred_enc = y_pred_wisard.copy()

    print("\nMelhor configuração encontrada para a WiSARD")
    print(f"N_BITS = {best_conf[0]}, tuple_size = {best_conf[1]}, Acurácia = {best_acc * 100:.2f}%")
 

    print("\nTreinamento Random Forest")
    acc_rf, y_pred_rf = train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test)
    print(f"Acurácia Random Forest: {acc_rf * 100:.2f}%")

    print("\nTreinamento SVM (LinearSVC)")
    acc_svm, y_pred_svm = train_svm(X_train_scaled, y_train, X_test_scaled, y_test)
    print(f"Acurácia SVM: {acc_svm * 100:.2f}%")

    print("\nTreinando KNN")
    acc_knn, y_pred_knn = train_knn(X_train_scaled, y_train, X_test_scaled, y_test)
    print(f"Acurácia KNN: {acc_knn * 100:.2f}%")

    print("\nResumo final")
    print("Classes:", list(label_encoder.classes_))

    print(f"WiSARD (melhor config): {best_acc * 100:.2f}%")
    print(f"Melhor N_BITS = {best_conf[0]}, melhor tuple_size = {best_conf[1]}")

    print(f"Random Forest: {acc_rf * 100:.2f}%")
    print(f"SVM (Linear):  {acc_svm * 100:.2f}%")
    print(f"KNN:           {acc_knn * 100:.2f}%")

    #save_wisard_prediction_csvs(X_test_df, y_test, best_y_pred_enc, label_encoder)

    save_prediction_distribution_plot(best_y_pred_enc, label_encoder, "WiSARD")
    save_prediction_distribution_plot(y_pred_rf, label_encoder, "Random Forest")
    save_prediction_distribution_plot(y_pred_svm, label_encoder, "SVM")
    save_prediction_distribution_plot(y_pred_knn, label_encoder, "KNN")

    save_wisardheatmap(results_wisard)
    save_wisardaccuracycurves(results_wisard)

    print("\nGerando matrizes de confusão...")

    save_confusion_matrix(y_test, best_y_pred_enc, label_encoder, "WiSARD")

    save_confusion_matrix(y_test, y_pred_rf, label_encoder, "Random Forest")

    save_confusion_matrix(y_test, y_pred_svm, label_encoder, "SVM")

    save_confusion_matrix(y_test, y_pred_knn, label_encoder, "KNN")

    save_combined_confusion_matrices(
    y_test,
    best_y_pred_enc,
    y_pred_rf,
    y_pred_svm,
    y_pred_knn,
    label_encoder
    )

    print("\nGerando CSVs separados por classe predita...")

    save_class_split_predictions(X_test_df, y_test, best_y_pred_enc,
                                label_encoder, "WiSARD")

    save_class_split_predictions(X_test_df, y_test, y_pred_rf,
                                label_encoder, "RandomForest")

    save_class_split_predictions(X_test_df, y_test, y_pred_svm,
                                label_encoder, "SVM")

    save_class_split_predictions(X_test_df, y_test, y_pred_knn,
                                label_encoder, "KNN")

if __name__ == "__main__":
    main()
