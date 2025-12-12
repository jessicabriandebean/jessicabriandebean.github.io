# %% [markdown]
# # Product Analytics Deep Dive
# 
# **Author:** Your Name  
# **Date:** December 2024  
# **Objective:** Comprehensive analysis of product metrics, user behavior, and growth opportunities
# 
# ## Table of Contents
# 1. [Data Loading & Exploration](#data-loading)
# 2. [User Engagement Analysis](#engagement)
# 3. [Feature Adoption & Usage](#features)
# 4. [Retention & Cohort Analysis](#retention)
# 5. [Conversion Funnel Analysis](#funnel)
# 6. [Predictive Analytics](#predictive)
# 7. [Recommendations](#recommendations)

# %% [markdown]
# ## 1. Data Loading & Exploration

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
sns.set_context("notebook", font_scale=1.1)

print("✅ Libraries imported successfully")

# %%
# Load analytics engine
from product_analytics import ProductAnalytics

analytics = ProductAnalytics()
analytics.generate_sample_data(days=60, num_users=10000)

print(f"📊 Dataset Summary:")
print(f"   • Total Users: {len(analytics.users):,}")
print(f"   • Total Events: {len(analytics.events):,}")
print(f"   • Date Range: {analytics.events['timestamp'].min().date()} to {analytics.events['timestamp'].max().date()}")
print(f"   • Unique Features: {analytics.events['feature'].nunique()}")

# %%
# Data preview
print("\n👤 Users Sample:")
display(analytics.users.head())

print("\n🎬 Events Sample:")
display(analytics.events.head())

# %%
# Basic statistics
print("\n📊 USER STATISTICS")
print("="*60)
print(analytics.users.describe(include='all'))

print("\n📊 EVENT STATISTICS")
print("="*60)
print(analytics.events['event_type'].value_counts())

# %% [markdown]
# ## 2. User Engagement Analysis

# %% [markdown]
# ### 2.1 Daily Active Users (DAU) & Monthly Active Users (MAU)

# %%
dau_mau = analytics.calculate_dau_mau()

fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=('Daily Active Users', 'Stickiness (DAU/MAU Ratio)'),
    vertical_spacing=0.15
)

# DAU plot
fig.add_trace(
    go.Scatter(
        x=dau_mau['date'],
        y=dau_mau['dau'],
        mode='lines+markers',
        name='DAU',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ),
    row=1, col=1
)

# Stickiness plot
fig.add_trace(
    go.Scatter(
        x=dau_mau['date'],
        y=dau_mau['stickiness'],
        mode='lines+markers',
        name='Stickiness %',
        line=dict(color='#ff7f0e', width=2)
    ),
    row=2, col=1
)

# Add benchmark line for stickiness
fig.add_hline(y=20, line_dash="dash", line_color="green", row=2, col=1,
              annotation_text="Healthy Benchmark (20%)")

fig.update_xaxes(title_text="Date", row=2, col=1)
fig.update_yaxes(title_text="Users", row=1, col=1)
fig.update_yaxes(title_text="Stickiness (%)", row=2, col=1)

fig.update_layout(height=700, showlegend=False, title_text="User Activity Metrics")
fig.show()

# %%
# Statistical summary
print("📊 DAU/MAU SUMMARY")
print("="*60)
print(f"Average DAU: {dau_mau['dau'].mean():.0f}")
print(f"Peak DAU: {dau_mau['dau'].max():.0f}")
print(f"Minimum DAU: {dau_mau['dau'].min():.0f}")
print(f"\nAverage Stickiness: {dau_mau['stickiness'].mean():.2f}%")
print(f"Best Stickiness: {dau_mau['stickiness'].max():.2f}%")

if dau_mau['stickiness'].mean() > 20:
    print("\n✅ Excellent engagement! Stickiness above 20% benchmark")
elif dau_mau['stickiness'].mean() > 15:
    print("\n📊 Good engagement. Room for improvement to reach 20% benchmark")
else:
    print("\n⚠️ Engagement needs attention. Consider user retention initiatives")

# %% [markdown]
# ### 2.2 Engagement Score Analysis

# %%
engagement = analytics.calculate_engagement_score()

print(f"Total users analyzed: {len(engagement)}")
print(f"\nEngagement Level Distribution:")
print(engagement['engagement_level'].value_counts())

# %%
# Engagement distribution
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Histogram
axes[0].hist(engagement['engagement_score'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
axes[0].axvline(engagement['engagement_score'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {engagement["engagement_score"].mean():.1f}')
axes[0].axvline(engagement['engagement_score'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {engagement["engagement_score"].median():.1f}')
axes[0].set_xlabel('Engagement Score', fontsize=12)
axes[0].set_ylabel('Number of Users', fontsize=12)
axes[0].set_title('Engagement Score Distribution', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Box plot by level
engagement_level_order = ['Low', 'Medium', 'High', 'Very High']
sns.boxplot(data=engagement, x='engagement_level', y='engagement_score', 
            order=engagement_level_order, ax=axes[1], palette='Set2')
axes[1].set_xlabel('Engagement Level', fontsize=12)
axes[1].set_ylabel('Engagement Score', fontsize=12)
axes[1].set_title('Engagement Score by Level', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Top 20 most engaged users
print("\n🏆 TOP 20 MOST ENGAGED USERS")
print("="*60)
top_engaged = engagement.head(20)[['user_id', 'engagement_score', 'engagement_level', 'total_events', 'features_used']]
display(top_engaged)

# %% [markdown]
# ## 3. Feature Adoption & Usage

# %%
features = analytics.calculate_feature_adoption()

print("📊 FEATURE ADOPTION METRICS")
print("="*60)
display(features)

# %%
# Feature adoption visualization
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Adoption Rate by Feature', 'Events per User by Feature'),
    specs=[[{'type': 'bar'}, {'type': 'bar'}]]
)

# Adoption rate
fig.add_trace(
    go.Bar(
        y=features['feature'],
        x=features['adoption_rate'],
        orientation='h',
        marker_color='lightblue',
        text=features['adoption_rate'].apply(lambda x: f'{x:.1f}%'),
        textposition='outside',
        name='Adoption Rate'
    ),
    row=1, col=1
)

# Events per user
fig.add_trace(
    go.Bar(
        y=features['feature'],
        x=features['avg_events_per_user'],
        orientation='h',
        marker_color='lightcoral',
        text=features['avg_events_per_user'].apply(lambda x: f'{x:.1f}'),
        textposition='outside',
        name='Avg Events/User'
    ),
    row=1, col=2
)

fig.update_xaxes(title_text="Adoption Rate (%)", row=1, col=1)
fig.update_xaxes(title_text="Avg Events per User", row=1, col=2)
fig.update_yaxes(title_text="Feature", row=1, col=1)

fig.update_layout(height=500, showlegend=False, title_text="Feature Usage Analysis")
fig.show()

# %%
# Feature adoption matrix
feature_events = analytics.events.groupby(['feature', 'user_segment']).size().reset_index(name='events')
feature_matrix = feature_events.pivot(index='feature', columns='user_segment', values='events')

plt.figure(figsize=(10, 6))
sns.heatmap(feature_matrix, annot=True, fmt='g', cmap='YlOrRd', cbar_kws={'label': 'Event Count'})
plt.title('Feature Usage by User Segment', fontsize=14, fontweight='bold')
plt.xlabel('User Segment', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Retention & Cohort Analysis

# %%
retention = analytics.calculate_retention(cohort_periods=10)

print("📊 COHORT RETENTION ANALYSIS")
print("="*60)
print(f"Cohorts analyzed: {retention['cohort'].nunique()}")
print(f"Periods tracked: {retention['period'].max()}")

# %%
# Retention heatmap
retention_pivot = retention.pivot_table(
    index='cohort',
    columns='period',
    values='retention_rate',
    fill_value=0
)

plt.figure(figsize=(14, 8))
sns.heatmap(
    retention_pivot,
    annot=True,
    fmt='.1f',
    cmap='RdYlGn',
    center=50,
    cbar_kws={'label': 'Retention Rate (%)'},
    linewidths=0.5
)
plt.title('Cohort Retention Heatmap', fontsize=16, fontweight='bold')
plt.xlabel('Weeks Since Signup', fontsize=12)
plt.ylabel('Cohort (Signup Week)', fontsize=12)
plt.tight_layout()
plt.show()

# %%
# Average retention curve
avg_retention = retention.groupby('period')['retention_rate'].mean().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(avg_retention['period'], avg_retention['retention_rate'], 
         marker='o', linewidth=2, markersize=8, color='#1f77b4')
plt.axhline(y=40, color='green', linestyle='--', alpha=0.7, label='Good Retention (40%)')
plt.axhline(y=25, color='orange', linestyle='--', alpha=0.7, label='Acceptable (25%)')
plt.fill_between(avg_retention['period'], 0, avg_retention['retention_rate'], alpha=0.2)

plt.xlabel('Weeks Since Signup', fontsize=12)
plt.ylabel('Retention Rate (%)', fontsize=12)
plt.title('Average Retention Curve', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Key retention metrics
print("\n📊 KEY RETENTION METRICS")
print("="*60)
print(f"Week 1 Retention: {avg_retention.iloc[1]['retention_rate']:.1f}%")
print(f"Week 4 Retention: {avg_retention.iloc[4]['retention_rate']:.1f}%")
print(f"Week 8 Retention: {avg_retention.iloc[8]['retention_rate']:.1f}%")

week_4_retention = avg_retention.iloc[4]['retention_rate']
if week_4_retention > 40:
    print("\n✅ Excellent retention! Well above industry benchmarks")
elif week_4_retention > 25:
    print("\n📊 Good retention, but room for improvement")
else:
    print("\n⚠️ Retention needs significant improvement")

# %% [markdown]
# ## 5. Conversion Funnel Analysis

# %%
# Define funnel steps
funnel_steps = ['page_view', 'button_click', 'form_submit', 'purchase']
funnel = analytics.calculate_funnel(funnel_steps)

print("📊 CONVERSION FUNNEL")
print("="*60)
display(funnel)

# %%
# Funnel visualization
fig = go.Figure()

# Funnel chart
fig.add_trace(go.Funnel(
    name='Conversion Funnel',
    y=funnel['step'],
    x=funnel['users'],
    textposition="inside",
    textinfo="value+percent initial",
    marker=dict(
        color=['#1f77b4', '#82ca9d', '#ffc658', '#ff7c7c']
    )
))

fig.update_layout(
    title='Conversion Funnel Analysis',
    height=500
)

fig.show()

# %%
# Drop-off analysis
funnel['drop_off_pct'] = (funnel['drop_off'] / funnel['users'].iloc[0] * 100)

plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(funnel)), funnel['drop_off'], color='lightcoral', edgecolor='black', alpha=0.7)
plt.xlabel('Funnel Step', fontsize=12)
plt.ylabel('Users Lost', fontsize=12)
plt.title('User Drop-off by Funnel Step', fontsize=14, fontweight='bold')
plt.xticks(range(len(funnel)), [f"{row['step']}\n→ {row['step']}" for _, row in funnel.iterrows() if _ < len(funnel)-1] + ['Final'], rotation=0)

# Add percentage labels
for i, (bar, val) in enumerate(zip(bars, funnel['drop_off_pct'])):
    if val > 0:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# %%
# Conversion rate summary
print("\n📊 CONVERSION INSIGHTS")
print("="*60)
overall_conversion = (funnel.iloc[-1]['users'] / funnel.iloc[0]['users'] * 100)
print(f"Overall Conversion Rate: {overall_conversion:.2f}%")
print(f"\nStep-by-step conversion:")
for i in range(len(funnel) - 1):
    step_conversion = (funnel.iloc[i+1]['users'] / funnel.iloc[i]['users'] * 100)
    print(f"  {funnel.iloc[i]['step']} → {funnel.iloc[i+1]['step']}: {step_conversion:.1f}%")

# %% [markdown]
# ## 6. User Segmentation Analysis

# %%
segments = analytics.calculate_user_segments()

segments_df = pd.DataFrame([
    {
        'Segment': k,
        'Users': v['count'],
        'Avg Events': v['avg_events_per_user'],
        'Top Feature': v['most_used_feature'],
        'Avg Session (sec)': v['avg_session_duration']
    }
    for k, v in segments.items()
])

print("📊 USER SEGMENTS ANALYSIS")
print("="*60)
display(segments_df)

# %%
# Segment comparison
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Avg Events per User', 'Avg Session Duration'),
    specs=[[{'type': 'bar'}, {'type': 'bar'}]]
)

fig.add_trace(
    go.Bar(
        x=segments_df['Segment'],
        y=segments_df['Avg Events'],
        marker_color='lightblue',
        name='Avg Events'
    ),
    row=1, col=1
)

fig.add_trace(
    go.Bar(
        x=segments_df['Segment'],
        y=segments_df['Avg Session (sec)'],
        marker_color='lightgreen',
        name='Avg Session Duration'
    ),
    row=1, col=2
)

fig.update_xaxes(title_text="User Segment", row=1, col=1)
fig.update_xaxes(title_text="User Segment", row=1, col=2)
fig.update_yaxes(title_text="Events per User", row=1, col=1)
fig.update_yaxes(title_text="Duration (seconds)", row=1, col=2)

fig.update_layout(height=500, showlegend=False, title_text="User Segment Comparison")
fig.show()

# %% [markdown]
# ## 7. Churn Risk Analysis

# %%
churned = analytics.detect_churned_users(inactive_days=30)

print(f"🚨 CHURN RISK ANALYSIS")
print("="*60)
print(f"Users at risk: {len(churned)} ({len(churned)/len(analytics.users)*100:.1f}% of total)")
print(f"\nChurn by segment:")
print(churned['user_segment'].value_counts())
print(f"\nChurn by plan:")
print(churned['plan'].value_counts())

# %%
# Churn visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# By segment
churn_segment = churned['user_segment'].value_counts()
axes[0].pie(churn_segment.values, labels=churn_segment.index, autopct='%1.1f%%', startangle=90)
axes[0].set_title('Churned Users by Segment', fontsize=14, fontweight='bold')

# By plan
churn_plan = churned['plan'].value_counts()
axes[1].pie(churn_plan.values, labels=churn_plan.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff', '#99ff99'])
axes[1].set_title('Churned Users by Plan', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# %%
# Days inactive distribution
plt.figure(figsize=(12, 6))
plt.hist(churned['days_inactive'], bins=30, color='salmon', edgecolor='black', alpha=0.7)
plt.axvline(churned['days_inactive'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {churned["days_inactive"].mean():.0f} days')
plt.xlabel('Days Inactive', fontsize=12)
plt.ylabel('Number of Users', fontsize=12)
plt.title('Distribution of Inactivity Period', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Key Insights & Recommendations

# %%
print("="*70)
print("KEY INSIGHTS & ACTIONABLE RECOMMENDATIONS")
print("="*70)

insights = analytics.generate_insights()

for i, (key, value) in enumerate(insights.items(), 1):
    print(f"\n{i}. {key.replace('_', ' ').upper()}")
    print(f"   {value}")

# %%
# Additional strategic insights
print("\n" + "="*70)
print("STRATEGIC RECOMMENDATIONS")
print("="*70)

recommendations = []

# Engagement
avg_stickiness = dau_mau['stickiness'].mean()
if avg_stickiness < 20:
    recommendations.append("📊 ENGAGEMENT: Implement daily habit loops and push notifications to improve DAU/MAU ratio")

# Retention
week_4_retention = avg_retention.iloc[4]['retention_rate']
if week_4_retention < 35:
    recommendations.append("🔄 RETENTION: Focus on onboarding improvements and early-stage user success programs")

# Feature adoption
low_adoption_features = features[features['adoption_rate'] < 30]
if len(low_adoption_features) > 0:
    recommendations.append(f"🎯 FEATURES: {len(low_adoption_features)} features have <30% adoption - consider in-app tutorials or feature discovery prompts")

# Churn
churn_rate = (len(churned) / len(analytics.users) * 100)
if churn_rate > 15:
    recommendations.append("🚨 CHURN: High churn rate - implement win-back campaigns and user feedback surveys")

# Conversion
if overall_conversion < 5:
    recommendations.append("💰 CONVERSION: Optimize checkout flow and reduce friction points in conversion funnel")

for i, rec in enumerate(recommendations, 1):
    print(f"\n{i}. {rec}")

# %% [markdown]
# ## 9. Executive Summary

# %%
print("\n" + "="*70)
print("EXECUTIVE SUMMARY")
print("="*70)

summary = {
    'Total Users': f"{len(analytics.users):,}",
    'Active Users (Last 30 Days)': f"{dau_mau['dau'].tail(30).mean():.0f}",
    'Stickiness (DAU/MAU)': f"{avg_stickiness:.1f}%",
    'Average Engagement Score': f"{engagement['engagement_score'].mean():.1f}/100",
    'Week 4 Retention': f"{week_4_retention:.1f}%",
    'Churn Rate': f"{churn_rate:.1f}%",
    'Top Feature': features.iloc[0]['feature'],
    'Overall Conversion Rate': f"{overall_conversion:.2f}%"
}

for key, value in summary.items():
    print(f"{key:.<40} {value}")

print("\n" + "="*70)

# %%
# Export summary
summary_df = pd.DataFrame([summary])
summary_df.to_csv('product_analytics_summary.csv', index=False)
print("\n✅ Summary exported to 'product_analytics_summary.csv'")

print("\n🎉 Analysis Complete!")
print("Next steps:")
print("  1. Review insights with product team")
print("  2. Prioritize recommendations by impact")
print("  3. Implement tracking for new metrics")
print("  4. Schedule follow-up analysis in 30 days")