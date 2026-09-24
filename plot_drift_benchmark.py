import numpy as np
import matplotlib.pyplot as plt

# شبیه‌سازی توزیع داده‌های پایه و شیفت‌یافته
np.random.seed(42)
baseline_feature = np.random.normal(0, 1, 2000)
drifted_feature = np.random.normal(1.8, 1.4, 2000)

# شبیه‌سازی p-value در ۱۰ پنجره زمانی متوالی تولید
windows = [f"W{i}" for i in range(1, 11)]
p_values = [0.82, 0.76, 0.65, 0.45, 0.28, 0.12, 0.04, 0.008, 0.001, 0.0002]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# نمودار ۱: Data Distribution Shift
ax1.hist(baseline_feature, bins=35, density=True, alpha=0.6, color='#2ecc71', edgecolor='black', label='Baseline Distribution (Training)')
ax1.hist(drifted_feature, bins=35, density=True, alpha=0.6, color='#e74c3c', edgecolor='black', label='Production Distribution (Drifted)')
ax1.set_xlabel('Feature Magnitude Value', fontsize=11, fontweight='bold')
ax1.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
ax1.set_title('A. Covariate Shift Detection (Kolmogorov-Smirnov)', fontsize=12, fontweight='bold', pad=12)
ax1.legend(frameon=True, loc='upper right')

# نمودار ۲: p-value Tracking over Time
ax2.plot(windows, p_values, color='#007acc', marker='o', linewidth=2.5, markersize=8, label='KS-Test p-value')
ax2.axhline(0.05, color='#e74c3c', linestyle='--', linewidth=2, label='Alert Threshold (α = 0.05)')
ax2.fill_between(windows, 0, 0.05, color='#e74c3c', alpha=0.15, label='Critical Drift Zone')
ax2.set_xlabel('Production Sliding Windows', fontsize=11, fontweight='bold')
ax2.set_ylabel('Statistical p-value', fontsize=11, fontweight='bold')
ax2.set_title('B. Continuous Drift Monitoring & Alert Triggering', fontsize=12, fontweight='bold', pad=12)
ax2.set_ylim(-0.05, 1.0)
ax2.legend(frameon=True, loc='upper right')

# نشان‌گذاری زمان دقیق آلارم
ax2.annotate('Drift Alarm\nTriggered (W7)', xy=(6, 0.04), xytext=(4.5, 0.35),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=8),
             fontweight='bold', fontsize=9.5, bbox=dict(boxstyle="round,pad=0.3", fc="#ffeeba", ec="#b8860b"))

plt.tight_layout()
output_filename = "mlops_drift_monitoring.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"MLOps monitoring figure successfully generated: {output_filename}")
