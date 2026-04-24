import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ── Step 1: Load CSV ─────────────────────────────────────────────
df = pd.read_csv('mercedes_benz_sales_2020_2025.csv')
df.columns = df.columns.str.strip()
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)

# ══════════════════════════════════════════════════════════════════
#  PART 1 — REVENUE PREDICTION
# ══════════════════════════════════════════════════════════════════

# ── Step 2: Create Revenue column ───────────────────────────────
df['Revenue'] = df['Base Price (USD)'] * df['Sales Volume']
print("\nRevenue column created successfully!")

# ── Step 3: Aggregate by Model + Year ───────────────────────────
agg = df.groupby(['Model', 'Year']).agg(
    Total_Revenue=('Revenue', 'sum'),
    Total_Sales=('Sales Volume', 'sum'),
    Avg_Price=('Base Price (USD)', 'mean'),
    Avg_HP=('Horsepower', 'mean')
).reset_index()

# ── Step 4: Encode Model column ──────────────────────────────────
le_model = LabelEncoder()
agg['Model_enc'] = le_model.fit_transform(agg['Model'])

# ── Step 5: Train / Test split ───────────────────────────────────
train = agg[agg['Year'] < 2025]
test  = agg[agg['Year'] == 2025]

features = ['Model_enc', 'Year', 'Avg_Price', 'Avg_HP']

X_train, y_train = train[features], train['Total_Revenue']
X_test,  y_test  = test[features],  test['Total_Revenue']

# ── Step 6: Train Revenue Model ──────────────────────────────────
revenue_model = RandomForestRegressor(n_estimators=100, random_state=42)
revenue_model.fit(X_train, y_train)
print("\nRevenue model trained!")

# ── Step 7: Evaluate Revenue Model ───────────────────────────────
rev_predictions = revenue_model.predict(X_test)
print("MAE:      $", round(mean_absolute_error(y_test, rev_predictions), 2))
print("R² Score:", round(r2_score(y_test, rev_predictions), 4))

# ── Step 8: Forecast Revenue 2026 & 2027 ────────────────────────
print("\n=== Revenue Forecast ===")
for forecast_year in [2026, 2027]:
    future_rev = agg[agg['Year'] == 2025][['Model', 'Model_enc', 'Avg_Price', 'Avg_HP']].copy()
    future_rev['Year'] = forecast_year
    future_rev['Predicted_Revenue'] = revenue_model.predict(future_rev[features]).astype(int)
    result_rev = future_rev[['Model', 'Predicted_Revenue']].sort_values(
        'Predicted_Revenue', ascending=False
    )
    print(f"\n{forecast_year} Revenue Forecast:")
    print(result_rev.to_string(index=False))

# Keep 2026 result for chart
future_2026 = agg[agg['Year'] == 2025][['Model', 'Model_enc', 'Avg_Price', 'Avg_HP']].copy()
future_2026['Year'] = 2026
future_2026['Predicted_Revenue'] = revenue_model.predict(future_2026[features]).astype(int)
result_2026 = future_2026[['Model', 'Predicted_Revenue']].sort_values(
    'Predicted_Revenue', ascending=False
)

future_2027 = agg[agg['Year'] == 2025][['Model', 'Model_enc', 'Avg_Price', 'Avg_HP']].copy()
future_2027['Year'] = 2027
future_2027['Predicted_Revenue'] = revenue_model.predict(future_2027[features]).astype(int)
result_2027 = future_2027[['Model', 'Predicted_Revenue']].sort_values(
    'Predicted_Revenue', ascending=False
)

# ══════════════════════════════════════════════════════════════════
#  PART 2 — FUEL TYPE PREDICTION
# ══════════════════════════════════════════════════════════════════

# ── Step 9: Aggregate fuel sales by Year ────────────────────────
fuel_agg = df.groupby(['Year', 'Fuel Type'])['Sales Volume'].sum().reset_index()
fuel_agg.columns = ['Year', 'Fuel_Type', 'Total_Sales']
print("\nFuel Sales by Year:")
print(fuel_agg)

# ── Step 10: Pivot fuel data ──────────────────────────────────────
pivot = fuel_agg.pivot(index='Year', columns='Fuel_Type', values='Total_Sales').fillna(0)
pivot.reset_index(inplace=True)
fuel_types = [c for c in pivot.columns if c != 'Year']

# ── Step 11: Build target — most popular fuel per year ───────────
pivot['Target'] = pivot[fuel_types].idxmax(axis=1)
le_fuel = LabelEncoder()
pivot['Target_enc'] = le_fuel.fit_transform(pivot['Target'])

X_fuel = pivot[['Year'] + fuel_types]
y_fuel = pivot['Target_enc']

# ── Step 12: Train Fuel Classifier ───────────────────────────────
fuel_model = RandomForestClassifier(n_estimators=100, random_state=42)
fuel_model.fit(X_fuel, y_fuel)
print("\nFuel classifier trained!")

# ── Step 13: Predict 2026 & 2027 fuel preference ────────────────
last_year_fuel = pivot[pivot['Year'] == 2025][fuel_types].values[0]

future_fuel_rows = []
for yr in [2026, 2027]:
    row = [yr] + list(last_year_fuel)
    future_fuel_rows.append(row)

future_fuel_df = pd.DataFrame(future_fuel_rows, columns=['Year'] + fuel_types)

fuel_predictions = fuel_model.predict(future_fuel_df)
predicted_fuel_labels = le_fuel.inverse_transform(fuel_predictions)

print("\n=== Fuel Type Predictions ===")
for yr, label in zip([2026, 2027], predicted_fuel_labels):
    print(f"{yr} → Most popular fuel type: {label}")

# ── Step 14: Prediction probabilities ────────────────────────────
proba = fuel_model.predict_proba(future_fuel_df)
proba_df = pd.DataFrame(proba, columns=le_fuel.classes_, index=[2026, 2027])
print("\n=== Prediction Probabilities ===")
print(proba_df.round(2))

colors_fuel = {
    'Petrol':   '#378ADD',
    'Diesel':   '#73726c',
    'Hybrid':   '#1D9E75',
    'Electric': '#D4537E'
}

actual_fuel = fuel_agg.pivot(
    index='Year', columns='Fuel_Type', values='Total_Sales'
).fillna(0)

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Mercedes-Benz Sales Intelligence — 2020 to 2027 Forecast',
             fontsize=16, fontweight='bold', y=1.01)

# ── Chart 1: Annual Sales Volume ─────────────────────────────────
yearly_sales = df.groupby('Year')['Sales Volume'].sum()
axes[0, 0].bar(yearly_sales.index, yearly_sales.values / 1e6,
               color='steelblue', edgecolor='white')
axes[0, 0].set_title('Annual Sales Volume (2020–2025)')
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Units Sold (Millions)')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# ── Chart 2: 2026 Revenue Forecast by Model ──────────────────────
axes[0, 1].barh(result_2026['Model'],
                result_2026['Predicted_Revenue'] / 1e9,
                color='#185FA5', edgecolor='white')
axes[0, 1].set_title('2026 Predicted Revenue by Model (USD Billions)')
axes[0, 1].set_xlabel('Revenue (Billions USD)')
axes[0, 1].invert_yaxis()
axes[0, 1].grid(True, alpha=0.3, axis='x')

# ── Chart 3: Historical Fuel Trend ───────────────────────────────
for fuel in actual_fuel.columns:
    axes[1, 0].plot(actual_fuel.index, actual_fuel[fuel] / 1000,
                    marker='o', label=fuel,
                    color=colors_fuel.get(fuel, 'gray'), linewidth=2)
axes[1, 0].set_title('Historical Fuel Type Sales (2020–2025)')
axes[1, 0].set_xlabel('Year')
axes[1, 0].set_ylabel('Sales Volume (Thousands)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# ── Chart 4: Fuel Forecast Probabilities 2026 & 2027 ─────────────
proba_df.T.plot(kind='bar', ax=axes[1, 1],
                color=['#185FA5', '#1D9E75'], edgecolor='white')
axes[1, 1].set_title('Predicted Fuel Preference — 2026 & 2027')
axes[1, 1].set_xlabel('Fuel Type')
axes[1, 1].set_ylabel('Probability')
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=30)
axes[1, 1].legend(title='Year')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('mercedes_full_forecast.png', dpi=150)
plt.show()
print("\nFull chart saved as mercedes_full_forecast.png")