import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc

# Mục đích script:
# - Đọc dữ liệu khách hàng từ Book3.csv
# - Xử lý dữ liệu thô và tạo các biến mới
# - Phân tích hành vi giữa khách ở lại và khách churn
# - Huấn luyện mô hình hồi quy logistic để dự đoán churn
# - Đánh giá mô hình bằng độ chính xác, ROC AUC, ma trận nhầm lẫn
# - Phân tích hệ số của mô hình để hiểu yếu tố ảnh hưởng

# Cấu hình UTF-8 để in tiếng Việt an toàn trong nhiều môi trường console
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

sns.set_theme(style='whitegrid')


def load_data(filepath: str) -> pd.DataFrame:
    """Đọc file CSV và trả về DataFrame."""
    return pd.read_csv(filepath, sep=';', encoding='utf-8', low_memory=False)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa dữ liệu và tạo các đặc trưng quan trọng."""
    for col in ['signup_date', 'last_purchase_date', 'order_date']:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    if 'customer_lifetime_days' not in df.columns:
        df['customer_lifetime_days'] = (df['last_purchase_date'] - df['signup_date']).dt.days
        df['customer_lifetime_days'] = df['customer_lifetime_days'].fillna(df['customer_lifetime_days'].median())

    if 'total_spend' not in df.columns:
        df['total_spend'] = df['unit_price'] * df['quantity']

    df['Churn'] = df['subscription_status'].apply(
        lambda x: 1 if str(x).strip().lower() == 'cancelled' else 0
    )
    return df


def explore_data(df: pd.DataFrame) -> None:
    """In báo cáo phân tích hành vi giữa nhóm churn và không churn."""
    behavior_summary = df.groupby('Churn').agg({
        'age': 'mean',
        'total_spend': 'mean',
        'purchase_frequence': 'mean',
        'quantity': 'mean',
        'customer_lifetime_days': 'mean'
    }).reset_index()
    print('So sánh hành vi trung bình giữa nhóm ở lại và nhóm rời bỏ:')
    print(behavior_summary.to_string(index=False))
    print('')


def prepare_features(df: pd.DataFrame):
    """Chuẩn bị ma trận đặc trưng và mảng nhãn mục tiêu."""
    numeric_features = [
        'age',
        'unit_price',
        'quantity',
        'purchase_frequence',
        'customer_lifetime_days',
        'total_spend'
    ]
    categorical_features = ['country', 'gender', 'category']

    X_num = df[numeric_features]
    X_cat = pd.get_dummies(df[categorical_features], drop_first=True)
    X = pd.concat([X_num, X_cat], axis=1)
    y = df['Churn']

    X_num_array = X_num.to_numpy()
    X_cat_array = X_cat.to_numpy()
    X_array = np.hstack([X_num_array, X_cat_array])
    y_array = y.to_numpy()

    feature_names = np.concatenate([X_num.columns.to_numpy(), X_cat.columns.to_numpy()])

    print('Số lượng feature sau mã hóa:', X.shape[1])
    print('Hình dạng mảng số:', X_num_array.shape)
    print('Hình dạng mảng phân loại đã mã hóa:', X_cat_array.shape)
    print('Hình dạng mảng đầu vào X:', X_array.shape)
    print('Hình dạng mảng mục tiêu y:', y_array.shape)
    print('')

    return X_array, y_array, feature_names


def split_and_scale(X_array: np.ndarray, y_array: np.ndarray):
    """Chia dữ liệu train/test và chuẩn hóa feature."""
    X_train, X_test, y_train, y_test = train_test_split(
        X_array, y_array, test_size=0.2, random_state=42, stratify=y_array
    )
    print('Kích thước train/test:', X_train.shape, X_test.shape)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print('Chuẩn hóa dữ liệu số hoàn tất.')
    print('')

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_model(X_train_scaled: np.ndarray, y_train: np.ndarray):
    """Huấn luyện mô hình Logistic Regression."""
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    return model


def train_random_baseline(X_train_scaled: np.ndarray, y_train: np.ndarray):
    """Huấn luyện mô hình baseline random để so sánh."""
    baseline = DummyClassifier(strategy='uniform', random_state=42)
    baseline.fit(X_train_scaled, y_train)
    return baseline


def evaluate_baseline(model, X_test_scaled: np.ndarray, y_test: np.ndarray):
    """Đánh giá mô hình baseline random và trả về các chỉ số chính."""
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    print('--- Mô hình baseline ngẫu nhiên ---')
    print('Baseline ngẫu nhiên dự đoán churn theo phân bố đều.')
    print('Độ chính xác:', round(accuracy, 4))
    print('ROC AUC:', round(roc_auc, 4))
    print('Báo cáo phân loại:')
    print(classification_report(y_test, y_pred))
    print('Ma trận nhầm lẫn:')
    print(confusion_matrix(y_test, y_pred))
    print('')

    return accuracy, roc_auc


def compare_models(logistic_metrics, baseline_metrics):
    """In so sánh trực tiếp giữa mô hình Logistic Regression và baseline random."""
    log_acc, log_auc = logistic_metrics
    base_acc, base_auc = baseline_metrics

    print('--- So sánh mô hình ---')
    print(f'Hồi quy logistic     | Độ chính xác = {log_acc:.4f}, ROC AUC = {log_auc:.4f}')
    print(f'Baseline ngẫu nhiên  | Độ chính xác = {base_acc:.4f}, ROC AUC = {base_auc:.4f}')

    if log_acc >= base_acc and log_auc >= base_auc:
        print('Kết luận: Logistic Regression vượt trội hơn baseline random trên cả accuracy và ROC AUC.')
    elif log_acc >= base_acc:
        print('Logistic Regression có accuracy tốt hơn, nhưng ROC AUC chưa chắc vượt qua random baseline.')
    elif log_auc >= base_auc:
        print('Logistic Regression có ROC AUC tốt hơn, nhưng accuracy chưa chắc vượt qua random baseline.')
    else:
        print('Kết luận: Logistic Regression hiện tại không vượt qua benchmark random baseline.')
    print('')


def save_evaluation_plots(y_test, y_pred, fpr, tpr, roc_auc: float) -> None:
    """Lưu biểu đồ confusion matrix và ROC curve vào file."""
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        annot=True,
        fmt='d',
        cmap='Greys',
        xticklabels=['Dự đoán Ở lại', 'Dự đoán Rời bỏ'],
        yticklabels=['Thực tế Ở lại', 'Thực tế Rời bỏ']
    )
    plt.title('Ma trận nhầm lẫn')

    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, color='black', lw=2, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Đường ROC')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('churn_evaluation.png', dpi=150)
    plt.show()
    print('Đã lưu biểu đồ đánh giá: churn_evaluation.png')
    print('')


def evaluate_model(model, X_test_scaled: np.ndarray, y_test: np.ndarray):
    """Đánh giá mô hình và vẽ biểu đồ kết quả."""
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    print('Độ chính xác:', round(accuracy, 4))
    print('ROC AUC:', round(roc_auc, 4))
    print('\nBáo cáo phân loại:')
    print(classification_report(y_test, y_pred))
    print('Ma trận nhầm lẫn:')
    print(confusion_matrix(y_test, y_pred))
    print('')

    save_evaluation_plots(y_test, y_pred, fpr, tpr, roc_auc)

    return accuracy, roc_auc


def save_coefficients_plot(coef_df: pd.DataFrame) -> None:
    """Lưu biểu đồ hệ số hồi quy logistic."""
    plt.figure(figsize=(10, 8))
    sns.barplot(data=coef_df, x='coef', y='feature')
    plt.axvline(0, color='black', linestyle='--')
    plt.title('Hệ số hồi quy logistic của các đặc trưng')
    plt.xlabel('Hệ số')
    plt.ylabel('Thuộc tính')
    plt.tight_layout()
    plt.savefig('churn_feature_importance.png', dpi=150)
    plt.show()
    print('Đã lưu biểu đồ trọng số: churn_feature_importance.png')


def analyze_coefficients(model, feature_names):
    """In và vẽ hệ số trọng số của Logistic Regression."""
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coef': model.coef_[0]
    }).sort_values(by='coef', ascending=False)

    print('Top 15 yếu tố ảnh hưởng lớn nhất:')
    print(coef_df.head(15).to_string(index=False))

    save_coefficients_plot(coef_df)


def main():
    print('--- 1. Đọc dữ liệu và tiền xử lý ---')
    print('Bước này đọc file CSV và chuẩn hóa các cột ngày tháng.')

    df = load_data('Book3.csv')
    print('Kích thước ban đầu:', df.shape)

    df = preprocess_data(df)
    print('Tạo biến mục tiêu Churn...')
    print('Tỷ lệ Churn (%):')
    print(df['Churn'].value_counts(normalize=True) * 100)
    print('')

    print('--- 2. Phân tích hành vi khách hàng ---')
    print('Bước này so sánh trung bình các chỉ số giữa khách churn và không churn.')
    explore_data(df)

    print('--- 3. Chuẩn bị dữ liệu cho mô hình ---')
    print('Bước này chọn đặc trưng, mã hóa biến phân loại và chuẩn hóa dữ liệu.')
    X_array, y_array, feature_names = prepare_features(df)
    X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale(X_array, y_array)

    print('--- 4. Huấn luyện Logistic Regression ---')
    print('Logistic Regression phù hợp cho bài toán phân lớp nhị phân và cho biết độ ảnh hưởng của từng biến.')
    model = train_model(X_train_scaled, y_train)

    print('--- 5. Huấn luyện baseline ngẫu nhiên ---')
    print('Baseline random dùng phân phối đều để dự đoán churn, nhằm so sánh hiệu quả thực tế.')
    baseline_model = train_random_baseline(X_train_scaled, y_train)
    baseline_accuracy, baseline_auc = evaluate_baseline(baseline_model, X_test_scaled, y_test)

    print('--- 6. Đánh giá Logistic Regression ---')
    print('Bước này đo lường hiệu suất và kiểm tra model có phân biệt được churn hay không.')
    logistic_accuracy, logistic_auc = evaluate_model(model, X_test_scaled, y_test)

    compare_models((logistic_accuracy, logistic_auc), (baseline_accuracy, baseline_auc))

    print('--- 7. Phân tích trọng số yếu tố ---')
    print('Hệ số của Logistic Regression cho biết hướng và độ mạnh ảnh hưởng của mỗi feature.')
    analyze_coefficients(model, feature_names)


if __name__ == '__main__':
    main()

