#!/usr/bin/env python
"""
Database migration script to set user_id for existing sales records.
Run this after updating the database schema to populate user_id for sales that were created before user tracking.
"""

import os
import sys
from app import create_app, db
from app.models import User, Sale

def migrate_sales_data():
    """
    Assign existing sales without user_id to the first admin user.
    If no admin user exists, the script will fail with a helpful message.
    """
    app = create_app()
    
    with app.app_context():
        print("🔍 Checking database schema...")
        
        # Get first admin user
        admin_user = User.query.filter_by(role='admin').first()
        
        if not admin_user:
            print("❌ No admin user found in the database.")
            print("   Please create an admin user first by registering through the web interface.")
            print("   The first registered user will automatically become an admin.")
            return False
        
        print(f"✓ Found admin user: {admin_user.username} (ID: {admin_user.id})")
        
        # Find sales without user_id
        orphaned_sales = Sale.query.filter_by(user_id=None).count()
        
        if orphaned_sales == 0:
            print("✓ All sales records already have user_id assigned.")
            return True
        
        print(f"📊 Found {orphaned_sales} sales records without user_id.")
        print(f"   Assigning them to {admin_user.username}...")
        
        # Update all orphaned sales
        Sale.query.filter_by(user_id=None).update({'user_id': admin_user.id}, synchronize_session=False)
        db.session.commit()
        
        print(f"✓ Successfully assigned {orphaned_sales} sales to {admin_user.username}")
        return True

def reset_database():
    """
    Completely reset the database (useful for development).
    This will drop all tables and recreate them.
    """
    app = create_app()
    
    with app.app_context():
        print("⚠️  RESETTING DATABASE...")
        print("   This will delete ALL data.")
        confirm = input("   Type 'yes' to confirm: ")
        
        if confirm.lower() != 'yes':
            print("   Cancelled.")
            return False
        
        print("🗑️  Dropping all tables...")
        db.drop_all()
        
        print("🏗️  Creating all tables...")
        db.create_all()
        
        print("✓ Database reset complete.")
        return True

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--reset':
        reset_database()
    else:
        migrate_sales_data()
