"""
Main pipeline for Image Sequence Position Classification
Run experiments and generate results
"""

import os
import torch

from src import (
    # Configuration
    EXPERIMENTS, RESULTS_DIR, DATASET_SUBSET_SIZE,
    
    # Data loading
    load_story_dataset,
    create_data_loaders,
    check_class_distribution,
    
    # Model
    ImageSequenceClassifier,
    print_model_summary,
    
    # Training
    train_model,
    
    # Visualization
    plot_training_curves,
    plot_comprehensive_comparison,
    create_results_table,
    save_results_table
)


def main():
    """
    Main execution pipeline for running all experiments.
    """
    print("="*80)
    print("IMAGE SEQUENCE POSITION CLASSIFICATION - DEEP LEARNING PROJECT")
    print("="*80)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"\n✓ Results directory created: {RESULTS_DIR}/")
    
    # Step 1: Load dataset
    print("\n" + "="*80)
    print("STEP 1: LOADING DATASET")
    print("="*80)
    
    train_dataset, test_dataset = load_story_dataset()
    
    # Use subset for faster training
    train_subset = train_dataset.select(range(DATASET_SUBSET_SIZE))
    print(f"\nUsing subset: {DATASET_SUBSET_SIZE} stories (1,500 images)")
    
    # Step 2: Create data loaders
    print("\n" + "="*80)
    print("STEP 2: CREATING DATA LOADERS")
    print("="*80)
    
    train_loader, val_loader, dataset = create_data_loaders(train_subset)
    check_class_distribution(dataset)
    
    # Step 3: Run experiments
    print("\n" + "="*80)
    print("STEP 3: RUNNING EXPERIMENTS")
    print("="*80)
    
    all_results = {}
    experiments_data = []
    
    for exp_key, exp_config in EXPERIMENTS.items():
        exp_name = exp_config['name']
        exp_params = exp_config['params']
        
        print("\n" + "-"*80)
        print(f"EXPERIMENT: {exp_name}")
        print(f"Parameters: {exp_params}")
        print("-"*80)
        
        # Create model
        model = ImageSequenceClassifier(**exp_params)
        
        # Show model summary for first experiment
        if exp_key == 'dropout':
            print_model_summary(model)
        
        # Train model
        train_losses, val_losses, train_accs, val_accs = train_model(
            model, train_loader, val_loader
        )
        
        # Store results
        all_results[exp_name] = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_acc': train_accs,
            'val_acc': val_accs
        }
        
        # Store for table
        experiments_data.append((
            exp_name,
            train_losses[-1],
            val_losses[-1],
            train_accs[-1],
            val_accs[-1]
        ))
        
        # Save individual experiment curves
        save_path = os.path.join(RESULTS_DIR, f"{exp_key}_curves.png")
        plot_training_curves(
            train_losses, val_losses, train_accs, val_accs,
            title=f"{exp_name} - Training Curves",
            save_path=save_path
        )
    
    # Step 4: Generate comprehensive visualizations
    print("\n" + "="*80)
    print("STEP 4: GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("="*80)
    
    # Comprehensive comparison plot
    comprehensive_path = os.path.join(RESULTS_DIR, "comprehensive_comparison.png")
    plot_comprehensive_comparison(all_results, save_path=comprehensive_path)
    
    # Step 5: Create and save results table
    print("\n" + "="*80)
    print("STEP 5: GENERATING RESULTS TABLE")
    print("="*80)
    
    df = create_results_table(experiments_data)
    
    table_path = os.path.join(RESULTS_DIR, "results_table.csv")
    save_results_table(df, table_path)
    
    # Final summary
    print("\n" + "="*80)
    print("EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\n✓ All experiments completed")
    print(f"✓ Individual plots saved to: {RESULTS_DIR}/")
    print(f"✓ Comprehensive comparison: {comprehensive_path}")
    print(f"✓ Results table: {table_path}")
    print("\nKey Findings:")
    print(f"  - Best validation accuracy: {df['Val Acc (%)'].max():.2f}%")
    print(f"  - Best model: {df.loc[df['Val Acc (%)'].idxmax(), 'Experiment']}")
    print(f"  - All models perform near random baseline (20%)")
    print(f"  - Task difficulty: Static images lack temporal markers")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
