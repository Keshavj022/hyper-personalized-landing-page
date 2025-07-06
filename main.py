import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

os.makedirs('images', exist_ok=True)

class HyperPersonalizedLandingPageGenerator:
    def __init__(self):
        self.user_segments = {}
        self.product_popularity = {}
        self.category_trends = {}
        self.cold_start_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def load_and_preprocess_data(self, user_df, trans_df):
        """Load and preprocess the datasets"""
        print("🔄 Loading and preprocessing data...")

        user_df['source'] = user_df['source'].replace({'(none)': 'unknown'})
        user_df['medium'] = user_df['medium'].replace({'(none)': 'unknown'})
        user_df['city'] = user_df['city'].fillna('unknown')
        user_df['region'] = user_df['region'].fillna('unknown')
        user_df['country'] = user_df['country'].fillna('unknown')
        user_df['source'] = user_df['source'].fillna('unknown')
        user_df['medium'] = user_df['medium'].fillna('unknown')
        user_df['income_group'] = user_df['income_group'].fillna('unknown')
        user_df['eventDate'] = pd.to_datetime(user_df['eventDate'])
        user_df['eventTimestamp'] = pd.to_datetime(user_df['eventTimestamp'])
        user_df['purchase_revenue'] = user_df['purchase_revenue'].fillna(0)
        user_df['total_item_quantity'] = user_df['total_item_quantity'].fillna(0)
        user_df = user_df.rename(columns={'transaction_id': 'Transaction_ID'})
        trans_df['Transaction_ID'] = trans_df['Transaction_ID'].astype(str)
        trans_df['Date'] = pd.to_datetime(trans_df['Date'])

        merged_df = user_df.merge(trans_df, on='Transaction_ID', how='left')

        self.user_df = user_df
        self.trans_df = trans_df
        self.merged_df = merged_df

        print("✅ Data preprocessing completed!")
        return merged_df

    def analyze_data_patterns(self):
        """Analyze and visualize data patterns"""
        print("📊 Analyzing data patterns...")

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('E-commerce Data Analysis Dashboard', fontsize=16, fontweight='bold')

        event_counts = self.user_df['event_name'].value_counts()
        axes[0, 0].pie(event_counts.values, labels=event_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Event Distribution')

        source_counts = self.user_df['source'].value_counts().head(10)
        axes[0, 1].bar(range(len(source_counts)), source_counts.values)
        axes[0, 1].set_xticks(range(len(source_counts)))
        axes[0, 1].set_xticklabels(source_counts.index, rotation=45)
        axes[0, 1].set_title('Top 10 Traffic Sources')

        device_counts = self.user_df['category'].value_counts()
        axes[0, 2].bar(device_counts.index, device_counts.values)
        axes[0, 2].set_title('Device Category Distribution')

        age_counts = self.user_df['Age'].value_counts()
        axes[1, 0].bar(age_counts.index, age_counts.values)
        axes[1, 0].set_title('Age Group Distribution')
        axes[1, 0].tick_params(axis='x', rotation=45)

        category_revenue = self.trans_df.groupby('ItemCategory')['Item_revenue'].sum().sort_values(ascending=False)
        axes[1, 1].bar(range(len(category_revenue)), category_revenue.values)
        axes[1, 1].set_xticks(range(len(category_revenue)))
        axes[1, 1].set_xticklabels(category_revenue.index, rotation=45)
        axes[1, 1].set_title('Revenue by Product Category')

        country_counts = self.user_df['country'].value_counts().head(10)
        axes[1, 2].barh(range(len(country_counts)), country_counts.values)
        axes[1, 2].set_yticks(range(len(country_counts)))
        axes[1, 2].set_yticklabels(country_counts.index)
        axes[1, 2].set_title('Top 10 Countries')

        plt.tight_layout()
        save_plot('Dataset-Analysis')

        self._analyze_user_journey()
        self._analyze_conversion_patterns()

    def _analyze_user_journey(self):
        """Analyze user journey patterns"""
        print("🔍 Analyzing user journey patterns...")

        user_sessions = self.user_df.groupby('user_pseudo_id').agg({
            'event_name': 'count',
            'page_type': lambda x: x.nunique(),
            'purchase_revenue': 'sum',
            'eventDate': ['min', 'max']
        }).reset_index()

        user_sessions.columns = ['user_pseudo_id', 'total_events', 'unique_pages', 'total_revenue', 'first_visit', 'last_visit']
        user_sessions['session_duration_days'] = (user_sessions['last_visit'] - user_sessions['first_visit']).dt.days

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].hist(user_sessions['total_events'], bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_title('Distribution of Events per User')
        axes[0].set_xlabel('Number of Events')
        axes[0].set_ylabel('Frequency')

        axes[1].hist(user_sessions['unique_pages'], bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1].set_title('Distribution of Unique Pages per User')
        axes[1].set_xlabel('Number of Unique Pages')
        axes[1].set_ylabel('Frequency')

        revenue_users = user_sessions[user_sessions['total_revenue'] > 0]
        if len(revenue_users) > 0:
            axes[2].hist(revenue_users['total_revenue'], bins=30, alpha=0.7, color='green', edgecolor='black')
            axes[2].set_title('Revenue Distribution (Paying Users)')
            axes[2].set_xlabel('Total Revenue')
            axes[2].set_ylabel('Frequency')

        plt.tight_layout()
        save_plot('User-Journey')

        self.user_sessions = user_sessions

    def _analyze_conversion_patterns(self):
        """Analyze conversion patterns"""
        print("📈 Analyzing conversion patterns...")

        funnel_events = ['session_start', 'page_view', 'view_item', 'add_to_cart', 'purchase']
        funnel_counts = []

        for event in funnel_events:
            count = self.user_df[self.user_df['event_name'] == event]['user_pseudo_id'].nunique()
            funnel_counts.append(count)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.barh(range(len(funnel_events)), funnel_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        ax1.set_yticks(range(len(funnel_events)))
        ax1.set_yticklabels(funnel_events)
        ax1.set_title('Conversion Funnel')
        ax1.set_xlabel('Number of Users')

        conversion_rates = [100]
        for i in range(1, len(funnel_counts)):
            if funnel_counts[0] > 0:
                rate = (funnel_counts[i] / funnel_counts[0]) * 100
                conversion_rates.append(rate)
            else:
                conversion_rates.append(0)

        ax2.plot(funnel_events, conversion_rates, marker='o', linewidth=2, markersize=8)
        ax2.set_title('Conversion Rates by Stage')
        ax2.set_ylabel('Conversion Rate (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_plot('Conversion-Patterns')

    def create_user_segments(self):
        """Create user segments based on behavior and demographics"""
        print("👥 Creating user segments...")

        user_features = self.user_df.groupby('user_pseudo_id').agg({
            'event_name': 'count',
            'page_type': lambda x: x.nunique(),
            'purchase_revenue': 'sum',
            'category': lambda x: x.mode()[0] if not x.empty else 'unknown',
            'gender': lambda x: x.mode()[0] if not x.empty else 'unknown',
            'Age': lambda x: x.mode()[0] if not x.empty else 'unknown',
            'income_group': lambda x: x.mode()[0] if not x.empty else 'unknown',
            'country': lambda x: x.mode()[0] if not x.empty else 'unknown',
            'source': lambda x: x.mode()[0] if not x.empty else 'unknown'
        }).reset_index()

        user_features.columns = ['user_pseudo_id', 'total_events', 'unique_pages', 'total_revenue',
                                'primary_device', 'gender', 'age', 'income_group', 'country', 'source']

        user_features['engagement_level'] = pd.cut(user_features['total_events'],
                                                  bins=[0, 5, 15, 50, float('inf')],
                                                  labels=['Low', 'Medium', 'High', 'Very High'])

        user_features['revenue_segment'] = pd.cut(user_features['total_revenue'],
                                                 bins=[-0.1, 0, 100, 500, float('inf')],
                                                 labels=['Non-buyer', 'Low-value', 'Medium-value', 'High-value'])

        categorical_cols = ['primary_device', 'gender', 'age', 'income_group', 'country', 'source']
        encoded_features = user_features.copy()

        for col in categorical_cols:
            le = LabelEncoder()
            encoded_features[col + '_encoded'] = le.fit_transform(encoded_features[col])
            self.label_encoders[col] = le

        cluster_features = ['total_events', 'unique_pages', 'total_revenue'] + \
                          [col + '_encoded' for col in categorical_cols]

        X = encoded_features[cluster_features].fillna(0)
        X_scaled = self.scaler.fit_transform(X)

        optimal_k = 6
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        user_features['cluster'] = kmeans.fit_predict(X_scaled)

        self._visualize_clusters(X_scaled, user_features, optimal_k)

        self.user_features = user_features
        self.kmeans_model = kmeans

        self._create_segment_profiles()

    def _visualize_clusters(self, X_scaled, user_features, optimal_k):
        """Visualize user clusters"""
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=user_features['cluster'], cmap='tab10', alpha=0.6)
        plt.title('User Clusters (PCA Visualization)')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.colorbar(scatter)

        plt.subplot(2, 2, 2)
        cluster_counts = user_features['cluster'].value_counts().sort_index()
        plt.bar(range(len(cluster_counts)), cluster_counts.values, color='skyblue')
        plt.title('User Distribution by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Users')

        plt.subplot(2, 2, 3)
        for cluster in range(optimal_k):
            cluster_data = user_features[user_features['cluster'] == cluster]
            plt.scatter(cluster_data['total_events'], cluster_data['total_revenue'],
                       label=f'Cluster {cluster}', alpha=0.6)
        plt.xlabel('Total Events')
        plt.ylabel('Total Revenue')
        plt.title('Engagement vs Revenue by Cluster')
        plt.legend()

        plt.subplot(2, 2, 4)
        cluster_summary = user_features.groupby('cluster')[['total_events', 'unique_pages', 'total_revenue']].mean()
        cluster_summary_normalized = (cluster_summary - cluster_summary.min()) / (cluster_summary.max() - cluster_summary.min())
        sns.heatmap(cluster_summary_normalized.T, annot=True, cmap='YlOrRd',
                   xticklabels=[f'Cluster {i}' for i in range(optimal_k)],
                   yticklabels=['Avg Events', 'Avg Pages', 'Avg Revenue'])
        plt.title('Cluster Characteristics (Normalized)')

        plt.tight_layout()
        save_plot('User-Segments')

    def _create_segment_profiles(self):
        """Create detailed segment profiles"""
        print("📋 Creating segment profiles...")

        segment_profiles = {}
        for cluster_id in self.user_features['cluster'].unique():
            cluster_data = self.user_features[self.user_features['cluster'] == cluster_id]

            profile = {
                'size': len(cluster_data),
                'avg_events': cluster_data['total_events'].mean(),
                'avg_pages': cluster_data['unique_pages'].mean(),
                'avg_revenue': cluster_data['total_revenue'].mean(),
                'top_device': cluster_data['primary_device'].mode()[0] if not cluster_data.empty else 'unknown',
                'top_gender': cluster_data['gender'].mode()[0] if not cluster_data.empty else 'unknown',
                'top_age': cluster_data['age'].mode()[0] if not cluster_data.empty else 'unknown',
                'top_income': cluster_data['income_group'].mode()[0] if not cluster_data.empty else 'unknown',
                'top_country': cluster_data['country'].mode()[0] if not cluster_data.empty else 'unknown',
                'top_source': cluster_data['source'].mode()[0] if not cluster_data.empty else 'unknown'
            }
            segment_profiles[f'Cluster_{cluster_id}'] = profile

        self.segment_profiles = segment_profiles

        profiles_df = pd.DataFrame(segment_profiles).T
        print("📊 Segment Profiles:")
        print(profiles_df.round(2))

    def analyze_product_trends(self):
        """Analyze product and category trends"""
        print("🛍️ Analyzing product trends...")

        category_popularity = self.trans_df.groupby('ItemCategory').agg({
            'Item_revenue': 'sum',
            'Item_purchase_quantity': 'sum',
            'Transaction_ID': 'nunique'
        }).reset_index()

        category_popularity.columns = ['Category', 'Total_Revenue', 'Total_Quantity', 'Unique_Transactions']
        category_popularity = category_popularity.sort_values('Total_Revenue', ascending=False)

        brand_popularity = self.trans_df.groupby('ItemBrand').agg({
            'Item_revenue': 'sum',
            'Item_purchase_quantity': 'sum'
        }).reset_index().sort_values('Item_revenue', ascending=False).head(10)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        axes[0, 0].bar(range(len(category_popularity)), category_popularity['Total_Revenue'])
        axes[0, 0].set_xticks(range(len(category_popularity)))
        axes[0, 0].set_xticklabels(category_popularity['Category'], rotation=45)
        axes[0, 0].set_title('Revenue by Product Category')
        axes[0, 0].set_ylabel('Total Revenue')

        axes[0, 1].bar(range(len(category_popularity)), category_popularity['Total_Quantity'], color='orange')
        axes[0, 1].set_xticks(range(len(category_popularity)))
        axes[0, 1].set_xticklabels(category_popularity['Category'], rotation=45)
        axes[0, 1].set_title('Quantity Sold by Category')
        axes[0, 1].set_ylabel('Total Quantity')

        axes[1, 0].barh(range(len(brand_popularity)), brand_popularity['Item_revenue'])
        axes[1, 0].set_yticks(range(len(brand_popularity)))
        axes[1, 0].set_yticklabels(brand_popularity['ItemBrand'])
        axes[1, 0].set_title('Top 10 Brands by Revenue')
        axes[1, 0].set_xlabel('Total Revenue')

        monthly_trends = self.trans_df.groupby(self.trans_df['Date'].dt.to_period('M'))['Item_revenue'].sum()
        axes[1, 1].plot(monthly_trends.index.astype(str), monthly_trends.values, marker='o')
        axes[1, 1].set_title('Monthly Revenue Trends')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Revenue')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        save_plot('Product-trends')

        self.category_popularity = category_popularity
        self.brand_popularity = brand_popularity

    def build_cold_start_model(self):
        """Build cold start recommendation model"""
        print("🚀 Building cold start recommendation model...")

        cold_start_features = ['category', 'gender', 'Age', 'income_group', 'country', 'source', 'medium']

        user_category_preference = self.user_df.groupby('user_pseudo_id').agg({
            'page_type': lambda x: x.mode()[0] if not x.empty else 'homepage',
            'category': lambda x: x.mode()[0] if not x.empty else 'unknown',
            'gender': lambda x: x.mode()[0] if not x.empty else 'unknown',
            'Age': lambda x: x.mode()[0] if not x.empty else 'unknown',
            'income_group': lambda x: x.mode()[0] if not x.empty else 'unknown',
            'country': lambda x: x.mode()[0] if not x.empty else 'unknown',
            'source': lambda x: x.mode()[0] if not x.empty else 'unknown',
            'medium': lambda x: x.mode()[0] if not x.empty else 'unknown'
        }).reset_index()

        for col in cold_start_features:
            if col not in self.label_encoders:
                le = LabelEncoder()
                user_category_preference[col + '_encoded'] = le.fit_transform(user_category_preference[col])
                self.label_encoders[col] = le
            else:
                user_category_preference[col + '_encoded'] = self.label_encoders[col].transform(user_category_preference[col])

        X = user_category_preference[[col + '_encoded' for col in cold_start_features]]
        y = user_category_preference['page_type']

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X, y)

        self.cold_start_model = rf_model
        self.cold_start_features = [col + '_encoded' for col in cold_start_features]

        feature_importance = pd.DataFrame({
            'feature': cold_start_features,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(feature_importance)), feature_importance['importance'])
        plt.yticks(range(len(feature_importance)), feature_importance['feature'])
        plt.title('Feature Importance for Cold Start Model')
        plt.xlabel('Importance')
        plt.tight_layout()
        save_plot('Feature-Importance')

        print("✅ Cold start model built successfully!")

    def generate_personalized_landing_page(self, user_profile):
        """Generate personalized landing page for a user"""
        print(f"🎯 Generating personalized landing page...")

        if hasattr(self, 'user_features') and 'user_pseudo_id' in user_profile and user_profile['user_pseudo_id'] in self.user_features['user_pseudo_id'].values:
            user_data = self.user_features[self.user_features['user_pseudo_id'] == user_profile['user_pseudo_id']].iloc[0]
            cluster = user_data['cluster']
            segment_profile = self.segment_profiles[f'Cluster_{cluster}']
            recommendations = self._get_cluster_recommendations(cluster)
        else:
            recommendations = self._get_cold_start_recommendations(user_profile)
            segment_profile = self._infer_segment_profile(user_profile)

        landing_page = {
            'hero_section': self._generate_hero_section(recommendations, segment_profile),
            'product_modules': self._generate_product_modules(recommendations),
            'cta_modules': self._generate_cta_modules(recommendations, segment_profile),
            'personalization_reason': recommendations.get('reason', 'Based on similar users')
        }

        return landing_page

    def _get_cluster_recommendations(self, cluster):
        """Get recommendations for a specific cluster"""
        cluster_users = self.user_features[self.user_features['cluster'] == cluster]['user_pseudo_id'].tolist()

        cluster_transactions = self.merged_df[self.merged_df['user_pseudo_id'].isin(cluster_users)]
        popular_categories = cluster_transactions.groupby('ItemCategory')['Item_revenue'].sum().sort_values(ascending=False)

        return {
            'top_categories': popular_categories.head(3).index.tolist(),
            'recommended_products': self._get_top_products(popular_categories.head(3).index.tolist()),
            'reason': f'Based on users in your segment (Cluster {cluster})'
        }

    def _get_cold_start_recommendations(self, user_profile):
        """Get recommendations for cold start scenario"""
        features = []
        cold_start_features = ['category', 'gender', 'Age', 'income_group', 'country', 'source', 'medium']

        for feature in cold_start_features:
            if feature in user_profile and feature in self.label_encoders:
                try:
                    encoded_value = self.label_encoders[feature].transform([user_profile[feature]])[0]
                except:
                    encoded_value = 0
                features.append(encoded_value)
            else:
                features.append(0)

        if hasattr(self, 'cold_start_model'):
            predicted_preference = self.cold_start_model.predict([features])[0]
        else:
            predicted_preference = 'homepage'

        similar_users = self.user_df[
            (self.user_df['gender'] == user_profile.get('gender', 'unknown')) &
            (self.user_df['Age'] == user_profile.get('Age', 'unknown')) &
            (self.user_df['country'] == user_profile.get('country', 'unknown'))
        ]

        if len(similar_users) > 0 and hasattr(self, 'category_popularity'):
            top_categories = self.category_popularity.head(3)['Category'].tolist()
        else:
            top_categories = ['Accessories', 'Footwear', 'Apparel']

        return {
            'top_categories': top_categories,
            'recommended_products': self._get_top_products(top_categories),
            'predicted_preference': predicted_preference,
            'reason': 'Based on similar users with your profile'
        }

    def _get_top_products(self, categories):
        """Get top products for given categories"""
        products = {}
        for category in categories:
            category_products = self.trans_df[self.trans_df['ItemCategory'] == category]
            if len(category_products) > 0:
                top_products = category_products.groupby('ItemName')['Item_revenue'].sum().sort_values(ascending=False).head(3)
                products[category] = top_products.index.tolist()
            else:
                products[category] = [f'Product 1 from {category}', f'Product 2 from {category}', f'Product 3 from {category}']
        return products

    def _infer_segment_profile(self, user_profile):
        """Infer segment profile for cold start users"""
        base_profile = {
            'avg_events': 10,
            'avg_pages': 5,
            'avg_revenue': 50,
            'top_device': user_profile.get('category', 'desktop'),
            'top_gender': user_profile.get('gender', 'unknown'),
            'top_age': user_profile.get('Age', 'unknown'),
            'top_income': user_profile.get('income_group', 'unknown'),
            'top_country': user_profile.get('country', 'unknown'),
            'top_source': user_profile.get('source', 'unknown')
        }
        return base_profile

    def _generate_hero_section(self, recommendations, segment_profile):
        """Generate hero section content"""
        top_category = recommendations['top_categories'][0] if recommendations['top_categories'] else 'Featured Products'

        hero_options = {
            'Accessories': {
                'title': 'Discover Premium Accessories',
                'subtitle': 'Elevate your style with our curated collection',
                'image': 'hero_accessories.jpg',
                'cta': 'Shop Accessories'
            },
            'Footwear': {
                'title': 'Step Into Comfort & Style',
                'subtitle': 'Find your perfect pair from our extensive collection',
                'image': 'hero_footwear.jpg',
                'cta': 'Shop Footwear'
            },
            'Apparel': {
                'title': 'Fashion That Fits Your Lifestyle',
                'subtitle': 'Discover clothing that speaks to you',
                'image': 'hero_apparel.jpg',
                'cta': 'Shop Apparel'
            }
        }

        return hero_options.get(top_category, {
            'title': 'Welcome to Our Store',
            'subtitle': 'Discover amazing products just for you',
            'image': 'hero_default.jpg',
            'cta': 'Start Shopping'
        })

    def _generate_product_modules(self, recommendations):
        """Generate product modules"""
        modules = []
        for category, products in recommendations.get('recommended_products', {}).items():
            module = {
                'title': f'Top {category}',
                'products': products,
                'layout': 'carousel' if len(products) > 2 else 'grid'
            }
            modules.append(module)
        return modules

    def _generate_cta_modules(self, recommendations, segment_profile):
        """Generate call-to-action modules"""
        cta_modules = []

        avg_events = segment_profile.get('avg_events', 10)
        avg_revenue = segment_profile.get('avg_revenue', 50)

        if avg_events < 5:
            cta_modules.append({
                'type': 'discovery',
                'title': 'Explore Our Collections',
                'message': 'New here? Discover what makes us special',
                'button_text': 'Start Exploring',
                'priority': 'high'
            })
        elif avg_events < 15:
            cta_modules.append({
                'type': 'engagement',
                'title': 'Find Your Perfect Match',
                'message': 'Browse our curated selections',
                'button_text': 'Shop Now',
                'priority': 'medium'
            })
        else:
            cta_modules.append({
                'type': 'loyalty',
                'title': 'Exclusive Member Benefits',
                'message': 'Join our loyalty program for special offers',
                'button_text': 'Join Now',
                'priority': 'high'
            })

        if avg_revenue == 0:
            cta_modules.append({
                'type': 'first_purchase',
                'title': 'Get 20% Off Your First Order',
                'message': 'Use code WELCOME20 at checkout',
                'button_text': 'Claim Offer',
                'priority': 'high'
            })
        elif avg_revenue < 100:
            cta_modules.append({
                'type': 'upsell',
                'title': 'Free Shipping on Orders $50+',
                'message': 'Add a few more items to qualify',
                'button_text': 'Continue Shopping',
                'priority': 'medium'
            })
        else:
            cta_modules.append({
                'type': 'vip',
                'title': 'VIP Early Access',
                'message': 'Get first access to new collections',
                'button_text': 'Join VIP',
                'priority': 'high'
            })

        return cta_modules

    def visualize_landing_page(self, landing_page, user_profile):
        """Visualize the generated landing page"""
        print("🎨 Visualizing personalized landing page...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Personalized Landing Page Layout', fontsize=16, fontweight='bold')

        # Hero Section Visualization
        ax1 = axes[0, 0]
        ax1.text(0.5, 0.8, landing_page['hero_section']['title'],
                ha='center', va='center', fontsize=14, fontweight='bold')
        ax1.text(0.5, 0.6, landing_page['hero_section']['subtitle'],
                ha='center', va='center', fontsize=10)
        ax1.text(0.5, 0.3, f"CTA: {landing_page['hero_section']['cta']}",
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_title('Hero Section')
        ax1.axis('off')

        # Product Modules
        ax2 = axes[0, 1]
        y_pos = 0.9
        for i, module in enumerate(landing_page['product_modules']):
            ax2.text(0.1, y_pos, f"• {module['title']}", fontsize=12, fontweight='bold')
            y_pos -= 0.1
            for product in module['products'][:3]:
                ax2.text(0.2, y_pos, f" - {product}", fontsize=10)
                y_pos -= 0.08
            y_pos -= 0.05
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_title('Product Modules')
        ax2.axis('off')

        # CTA Modules
        ax3 = axes[1, 0]
        y_pos = 0.9
        for cta in landing_page['cta_modules']:
            priority_color = {'high': 'red', 'medium': 'orange', 'low': 'green'}
            ax3.text(0.1, y_pos, f"• {cta['title']}", fontsize=12, fontweight='bold',
                    color=priority_color.get(cta['priority'], 'black'))
            y_pos -= 0.1
            ax3.text(0.2, y_pos, cta['message'], fontsize=10)
            y_pos -= 0.08
            ax3.text(0.2, y_pos, f"Button: {cta['button_text']}", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen"))
            y_pos -= 0.15
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.set_title('Call-to-Action Modules')
        ax3.axis('off')

        # User Profile Summary
        ax4 = axes[1, 1]
        profile_text = f"User Profile:\n"
        for key, value in user_profile.items():
            profile_text += f"• {key}: {value}\n"
        profile_text += f"\nPersonalization Reason:\n{landing_page['personalization_reason']}"
        ax4.text(0.1, 0.9, profile_text, fontsize=10, va='top')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.set_title('User Profile & Reasoning')
        ax4.axis('off')

        plt.tight_layout()
        save_plot('Landing-Page-Layout')

    def evaluate_recommendations(self):
        """Evaluate recommendation system performance"""
        print("📊 Evaluating recommendation system...")

        test_scenarios = [
            {'category': 'mobile', 'gender': 'Female', 'Age': '25-34', 'country': 'United States'},
            {'category': 'desktop', 'gender': 'Male', 'Age': '35-44', 'country': 'United Kingdom', 'source': 'google'},
            {'category': 'mobile', 'gender': 'Female', 'Age': '18-24', 'country': 'Canada', 'source': 'facebook'},
            {'category': 'tablet', 'gender': 'Male', 'Age': '45-54', 'country': 'Australia', 'source': 'direct'}
        ]

        recommendations_summary = []

        for i, scenario in enumerate(test_scenarios):
            print(f"\n--- Test Scenario {i+1} ---")
            landing_page = self.generate_personalized_landing_page(scenario)

            summary = {
                'scenario': i+1,
                'user_type': 'Existing' if 'user_pseudo_id' in scenario else 'New',
                'top_category': landing_page['product_modules'][0]['title'] if landing_page['product_modules'] else 'None',
                'num_cta_modules': len(landing_page['cta_modules']),
                'hero_title': landing_page['hero_section']['title'],
                'personalization_reason': landing_page['personalization_reason']
            }
            recommendations_summary.append(summary)

            self.visualize_landing_page(landing_page, scenario)

        summary_df = pd.DataFrame(recommendations_summary)
        print("\n📋 Recommendations Summary:")
        print(summary_df.to_string(index=False))

        return summary_df

    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        print("📈 Generating performance report...")

        if not hasattr(self, 'user_features'):
            print("Warning: User segmentation not performed. Running segmentation first...")
            self.create_user_segments()

        total_users = len(self.user_features)
        total_transactions = len(self.trans_df)
        avg_revenue_per_user = self.user_features['total_revenue'].mean()
        conversion_rate = (self.user_features['total_revenue'] > 0).mean() * 100

        segment_sizes = self.user_features['cluster'].value_counts().sort_index()
        segment_revenue = self.user_features.groupby('cluster')['total_revenue'].mean()

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Hyper-Personalized Landing Page Generator - Performance Report', fontsize=16, fontweight='bold')

        # System overview
        ax1 = axes[0, 0]
        metrics = ['Total Users', 'Total Transactions', 'Avg Revenue/User', 'Conversion Rate %']
        values = [total_users, total_transactions, avg_revenue_per_user, conversion_rate]
        bars = ax1.bar(range(len(metrics)), values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax1.set_xticks(range(len(metrics)))
        ax1.set_xticklabels(metrics, rotation=45)
        ax1.set_title('System Overview')

        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                    f'{value:.1f}', ha='center', va='bottom')

        # Segment distribution
        ax2 = axes[0, 1]
        ax2.pie(segment_sizes.values, labels=[f'Cluster {i}' for i in segment_sizes.index],
               autopct='%1.1f%%', startangle=90)
        ax2.set_title('User Segment Distribution')

        # Revenue by segment
        ax3 = axes[0, 2]
        bars = ax3.bar(range(len(segment_revenue)), segment_revenue.values, color='lightcoral')
        ax3.set_xticks(range(len(segment_revenue)))
        ax3.set_xticklabels([f'Cluster {i}' for i in segment_revenue.index])
        ax3.set_title('Average Revenue by Segment')
        ax3.set_ylabel('Average Revenue')

        # Category performance
        ax4 = axes[1, 0]
        if hasattr(self, 'category_popularity'):
            top_categories = self.category_popularity.head(5)
            ax4.barh(range(len(top_categories)), top_categories['Total_Revenue'])
            ax4.set_yticks(range(len(top_categories)))
            ax4.set_yticklabels(top_categories['Category'])
            ax4.set_title('Top 5 Categories by Revenue')
            ax4.set_xlabel('Total Revenue')

        # Cold start model performance
        ax5 = axes[1, 1]
        if hasattr(self, 'cold_start_model'):
            feature_names = ['Device', 'Gender', 'Age', 'Income', 'Country', 'Source', 'Medium']
            importance = self.cold_start_model.feature_importances_
            ax5.bar(range(len(importance)), importance)
            ax5.set_xticks(range(len(importance)))
            ax5.set_xticklabels(feature_names, rotation=45)
            ax5.set_title('Cold Start Model Feature Importance')
            ax5.set_ylabel('Importance')

        # Recommendation coverage
        ax6 = axes[1, 2]
        coverage_metrics = ['Categories Covered', 'Products Recommended', 'CTA Variations', 'Hero Themes']
        coverage_values = [
            len(self.category_popularity) if hasattr(self, 'category_popularity') else 0,
            self.trans_df['ItemName'].nunique(),
            8,
            4
        ]
        ax6.bar(range(len(coverage_metrics)), coverage_values, color='lightgreen')
        ax6.set_xticks(range(len(coverage_metrics)))
        ax6.set_xticklabels(coverage_metrics, rotation=45)
        ax6.set_title('Recommendation System Coverage')

        plt.tight_layout()
        save_plot('Performance-Report')

        print("\n📊 PERFORMANCE SUMMARY")
        print("=" * 50)
        print(f"Total Users Analyzed: {total_users:,}")
        print(f"Total Transactions: {total_transactions:,}")
        print(f"Average Revenue per User: ${avg_revenue_per_user:.2f}")
        print(f"Overall Conversion Rate: {conversion_rate:.2f}%")
        print(f"Number of User Segments: {len(segment_sizes)}")
        print(f"Categories Available: {len(self.category_popularity) if hasattr(self, 'category_popularity') else 0}")
        print(f"Unique Products: {self.trans_df['ItemName'].nunique():,}")

        return {
            'total_users': total_users,
            'total_transactions': total_transactions,
            'avg_revenue_per_user': avg_revenue_per_user,
            'conversion_rate': conversion_rate,
            'num_segments': len(segment_sizes),
            'num_categories': len(self.category_popularity) if hasattr(self, 'category_popularity') else 0,
            'num_products': self.trans_df['ItemName'].nunique()
        }


def save_plot(filename, dpi=300, format='png'):
    """Save current plot to images folder"""
    plt.savefig(f'images2/{filename}.{format}', 
                dpi=dpi, 
                bbox_inches='tight', 
                facecolor='white',
                edgecolor='none')
    plt.close()
    print(f'Plot Saved to images/{filename}.{format}')


def main():
    """Main function to demonstrate the landing page generator"""
    print("🚀 Starting Hyper-Personalized Landing Page Generator")
    print("=" * 60)

    # Load data - replace with your actual file paths
    try:
        user_df = pd.read_csv('data/dataset1_final.csv')
        trans_df = pd.read_csv('data/dataset2_final.csv')
    except FileNotFoundError:
        print("Please update the file paths to your actual data files")
        return None

    # Initialize the generator
    generator = HyperPersonalizedLandingPageGenerator()

    # Load and preprocess data
    generator.load_and_preprocess_data(user_df, trans_df)

    # Analyze data patterns
    generator.analyze_data_patterns()

    # Create user segments
    generator.create_user_segments()

    # Analyze product trends
    generator.analyze_product_trends()

    # Build cold start model
    generator.build_cold_start_model()

    # Test with sample user profile
    sample_user_profile = {
        'category': 'mobile',
        'gender': 'Female',
        'Age': '25-34',
        'income_group': 'medium',
        'country': 'United States',
        'source': 'google',
        'medium': 'organic'
    }

    print("\n🎯 Testing with sample user profile:")
    print(sample_user_profile)

    landing_page = generator.generate_personalized_landing_page(sample_user_profile)
    generator.visualize_landing_page(landing_page, sample_user_profile)

    # Evaluate recommendations
    generator.evaluate_recommendations()

    # Generate performance report
    generator.generate_performance_report()

    return generator

if __name__ == "__main__":
    generator = main()