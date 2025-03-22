import os
import json
import time
import gc
import psutil
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import asdict
import importlib.util
import concurrent.futures

from .metrics import (
    EvaluationMetric, BenchmarkConfig, ArchitectureType, 
    AgentType, ArchitectureMetrics, AgentMetrics
)
from .utils import (
    calculate_accuracy_score, calculate_domain_specific_accuracy,
    measure_inter_agent_communication, analyze_state_serialization,
    measure_adapter_overhead, analyze_accuracy_preservation,
    time_execution
)
from .visualizations import (
    plot_bar_chart, plot_multi_architecture_comparison,
    plot_radar_chart, plot_accuracy_preservation_heatmap,
    plot_performance_vs_accuracy, plot_timeline_visualization
)

class AgentFrameworkEvaluator:
    """Comprehensive evaluator for multi-agent, composable, and hybrid frameworks"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.architectures = {}  # Maps architecture name to implementation
        self.adapters = {}       # Maps framework name to adapter
        self.results = {}        # Maps architecture name to results
        self.ground_truth = self._load_ground_truth()
        self.baseline_results = None  # Will store multi-framework baseline results
        
        # Ensure results directory exists
        os.makedirs(self.config.results_path, exist_ok=True)
        
        # Load architectures and adapters
        self._load_architectures()
        self._load_adapters()

    def _load_ground_truth(self) -> Dict[str, Any]:
        """Load ground truth data for evaluation scenarios"""
        gt_path = self.config.ground_truth_path
        ground_truth = {}
        
        if not os.path.exists(gt_path):
            print(f"WARNING: Missing ground truth path: {gt_path}")
            return ground_truth
            
        for fname in os.listdir(gt_path):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(gt_path, fname), 'r') as f:
                        data = json.load(f)
                        if "scenario_id" in data:
                            ground_truth[data["scenario_id"]] = data
                except Exception as e:
                    print(f"Error loading ground truth file {fname}: {e}")
                    
        return ground_truth

    def _load_architectures(self):
        """Load architecture implementations"""
        arch_dir = os.path.join(os.path.dirname(__file__), "../architectures")
        
        if not os.path.exists(arch_dir):
            os.makedirs(arch_dir, exist_ok=True)
            print(f"Created architectures directory: {arch_dir}")
            return
            
        for arch_type in self.config.architectures:
            # Convert enum value to string if needed
            if isinstance(arch_type, ArchitectureType):
                arch_name = arch_type.value
            else:
                arch_name = arch_type
                
            arch_file = f"{arch_name}_architecture.py"
            path = os.path.join(arch_dir, arch_file)
            
            if not os.path.exists(path):
                print(f"Architecture implementation not found: {path}")
                continue
                
            try:
                spec = importlib.util.spec_from_file_location(arch_name, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                arch_class = getattr(module, f"{arch_name.capitalize()}Architecture")
                self.architectures[arch_name] = arch_class()
                print(f"Loaded architecture: {arch_name}")
            except Exception as e:
                print(f"Error loading architecture {arch_name}: {e}")

    def _load_adapters(self):
        """Load framework adapters"""
        adapter_dir = os.path.join(os.path.dirname(__file__), "../adapters")
        
        if not os.path.exists(adapter_dir):
            os.makedirs(adapter_dir, exist_ok=True)
            print(f"Created adapters directory: {adapter_dir}")
            return
            
        for fw_name in self.config.frameworks or []:
            adapter_file = f"{fw_name}_adapter.py"
            path = os.path.join(adapter_dir, adapter_file)
            
            if not os.path.exists(path):
                print(f"Adapter not found: {path}")
                continue
                
            try:
                spec = importlib.util.spec_from_file_location(fw_name, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                adapter_class = getattr(module, f"{fw_name.capitalize()}Adapter")
                self.adapters[fw_name] = adapter_class()
                print(f"Loaded adapter: {fw_name}")
            except Exception as e:
                print(f"Error loading adapter {fw_name}: {e}")

    def evaluate_all(self):
        """Evaluate all configured architectures"""
        # First evaluate multi-framework as baseline if available
        if "multi_framework" in self.architectures:
            print("Evaluating baseline multi-framework architecture...")
            self.baseline_results = self._evaluate_architecture("multi_framework")
            self.results["multi_framework"] = self.baseline_results
        
        # Then evaluate other architectures
        for arch_name in self.architectures:
            if arch_name != "multi_framework":  # Skip if already evaluated
                print(f"Evaluating {arch_name} architecture...")
                self.results[arch_name] = self._evaluate_architecture(arch_name)
        
        # Generate comparative visualizations
        self._generate_comparative_visualizations()
        
        return self.results

    def _evaluate_architecture(self, arch_name: str) -> Dict[str, Any]:
        """Evaluate a specific architecture implementation"""
        architecture = self.architectures[arch_name]
        
        # Metrics to collect
        metrics = {
            EvaluationMetric.EXECUTION_TIME.value: [],
            EvaluationMetric.MEMORY_USAGE.value: [],
            EvaluationMetric.RESPONSE_LATENCY.value: [],
            EvaluationMetric.COLD_START_TIME.value: None,
            EvaluationMetric.THROUGHPUT.value: None,
            EvaluationMetric.CPU_UTILIZATION.value: [],
        }
        
        # For multi-agent specific metrics
        if self.config.inter_agent_communication_tracking:
            metrics[EvaluationMetric.INTER_AGENT_LATENCY.value] = []
            metrics[EvaluationMetric.MESSAGE_OVERHEAD.value] = []
            
        # For composability metrics
        if self.config.state_serialization_tracking:
            metrics[EvaluationMetric.STATE_SERIALIZATION_OVERHEAD.value] = []
            
        # For hybrid architecture metrics
        if self.config.adapter_overhead_tracking:
            metrics[EvaluationMetric.ADAPTER_OVERHEAD.value] = []
            metrics[EvaluationMetric.FRAMEWORK_SWITCHING_LATENCY.value] = []
        
        # Accuracy results by category and scenario
        accuracy_results = {
            "overall_accuracy": 0.0,
            "by_category": {},
            "by_agent_type": {},
            "error_cases": []
        }
        
        # Group scenarios by category
        scenarios_by_category = {}
        for sid, scenario in self.ground_truth.items():
            category = scenario.get("category", "default")
            scenarios_by_category.setdefault(category, []).append(sid)
        
        # Measure cold start time
        cold_start_begin = time.time()
        architecture.initialize()
        cold_start_end = time.time()
        metrics[EvaluationMetric.COLD_START_TIME.value] = (cold_start_end - cold_start_begin) * 1000  # ms
        
        # Process each scenario
        total_scenarios = len(self.ground_truth)
        start_time = time.time()
        
        # Track communication and state events
        communication_events = []
        serialization_events = []
        adapter_events = []
        execution_timeline = []
        
        for sid, scenario in self.ground_truth.items():
            # Clear memory before each run
            gc.collect()
            
            # Record memory before
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Prepare input
            input_data = scenario["input"]
            expected_output = scenario["expected_output"]
            category = scenario.get("category", "default")
            agent_types = scenario.get("agent_types", [])
            
            # Execute scenario
            scenario_start = time.time()
            
            # Execute with event tracking
            actual_output, events = architecture.run_with_events(
                input_data,
                track_communication=self.config.inter_agent_communication_tracking,
                track_serialization=self.config.state_serialization_tracking,
                track_adapters=self.config.adapter_overhead_tracking
            )
            
            scenario_end = time.time()
            
            # Collect events
            if "communication" in events:
                communication_events.extend(events["communication"])
            if "serialization" in events:
                serialization_events.extend(events["serialization"])
            if "adapters" in events:
                adapter_events.extend(events["adapters"])
            if "timeline" in events:
                for event in events["timeline"]:
                    if "start_time" not in event:
                        event["start_time"] = scenario_start * 1000  # Convert to ms
                    execution_timeline.append(event)
            
            # Calculate response latency
            latency = (scenario_end - scenario_start) * 1000  # ms
            metrics[EvaluationMetric.RESPONSE_LATENCY.value].append(latency)
            
            # Record memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = memory_after - memory_before
            metrics[EvaluationMetric.MEMORY_USAGE.value].append(memory_usage)
            
            # Record CPU utilization
            cpu_percent = process.cpu_percent()
            metrics[EvaluationMetric.CPU_UTILIZATION.value].append(cpu_percent)
            
            # Calculate accuracy
            if agent_types and len(agent_types) > 0:
                # Calculate domain-specific accuracy for each agent type
                scores = []
                for agent_type in agent_types:
                    if agent_type in expected_output:
                        agent_expected = expected_output[agent_type]
                        agent_actual = actual_output.get(agent_type, {})
                        score = calculate_domain_specific_accuracy(
                            agent_actual, agent_expected, agent_type
                        )
                        scores.append(score)
                        
                        # Track by agent type
                        accuracy_results["by_agent_type"].setdefault(agent_type, []).append(score)
                
                # Average across agent types
                scenario_score = sum(scores) / len(scores) if scores else 0.0
            else:
                # Generic accuracy calculation
                scenario_score = calculate_accuracy_score(actual_output, expected_output)
            
            # Track accuracy by category
            accuracy_results["by_category"].setdefault(category, []).append(scenario_score)
            
            # Track errors
            if scenario_score < self.config.accuracy_threshold:
                accuracy_results["error_cases"].append({
                    "scenario_id": sid,
                    "category": category,
                    "score": scenario_score,
                    "agent_types": agent_types
                })
        
        # Calculate execution time
        end_time = time.time()
        total_execution_time = (end_time - start_time) * 1000  # ms
        metrics[EvaluationMetric.EXECUTION_TIME.value] = total_execution_time
        
        # Calculate throughput (scenarios per second)
        throughput = total_scenarios / (total_execution_time / 1000)
        metrics[EvaluationMetric.THROUGHPUT.value] = throughput
        
        # Process multi-agent metrics
        if communication_events:
            avg_latency, total_overhead, msg_count = measure_inter_agent_communication(communication_events)
            metrics[EvaluationMetric.INTER_AGENT_LATENCY.value] = avg_latency
            metrics[EvaluationMetric.MESSAGE_OVERHEAD.value] = total_overhead
        
        # Process serialization metrics
        if serialization_events:
            avg_time, avg_overhead = analyze_state_serialization(serialization_events)
            metrics[EvaluationMetric.STATE_SERIALIZATION_OVERHEAD.value] = avg_time
        
        # Process adapter metrics
        if adapter_events:
            avg_conversion, avg_execution, avg_switching = measure_adapter_overhead(adapter_events)
            metrics[EvaluationMetric.ADAPTER_OVERHEAD.value] = avg_conversion
            metrics[EvaluationMetric.FRAMEWORK_SWITCHING_LATENCY.value] = avg_switching
        
        # Calculate overall accuracy
        all_scores = []
        for category_scores in accuracy_results["by_category"].values():
            all_scores.extend(category_scores)
        
        accuracy_results["overall_accuracy"] = sum(all_scores) / len(all_scores) if all_scores else 0.0
        
        # Calculate average accuracy by category
        for category, scores in accuracy_results["by_category"].items():
            accuracy_results["by_category"][category] = sum(scores) / len(scores)
        
        # Calculate average accuracy by agent type
        for agent_type, scores in accuracy_results["by_agent_type"].items():
            accuracy_results["by_agent_type"][agent_type] = sum(scores) / len(scores)
        
        # Combine all results
        results = {
            "metrics": metrics,
            "accuracy": accuracy_results,
            "execution_timeline": execution_timeline
        }
        
        # If we have baseline results and this isn't the baseline, calculate preservation
        if self.baseline_results and arch_name != "multi_framework":
            baseline_category_accuracy = self.baseline_results["accuracy"]["by_category"]
            current_category_accuracy = accuracy_results["by_category"]
            
            preservation = analyze_accuracy_preservation(
                baseline_category_accuracy,
                current_category_accuracy
            )
            
            results["accuracy_preservation"] = preservation
            
            # Add to metrics for visualization
            metrics[EvaluationMetric.ACCURACY_PRESERVATION.value] = preservation.get("overall", 0.0)
        
        # Save results
        self._save_results(arch_name, results)
        
        # Generate visualizations
        self._generate_visualizations(arch_name, results)
        
        return results

    def _save_results(self, arch_name: str, results: Dict[str, Any]):
        """Save evaluation results to disk"""
        path = os.path.join(self.config.results_path, arch_name)
        os.makedirs(path, exist_ok=True)
        
        with open(os.path.join(path, "results.json"), 'w') as f:
            # Convert any non-serializable objects
            clean_results = self._make_serializable(results)
            json.dump(clean_results, f, indent=2)
        
        print(f"Results saved for {arch_name} at {path}/results.json")

    def _make_serializable(self, obj):
        """Recursively convert results to JSON-serializable format"""
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, 'to_dict'):
            return self._make_serializable(obj.to_dict())
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return str(obj)

    def _generate_visualizations(self, arch_name: str, results: Dict[str, Any]):
        """Generate visualizations for a single architecture's results"""
        viz_path = os.path.join(self.config.results_path, arch_name, "visualizations")
        os.makedirs(viz_path, exist_ok=True)
        
        # Accuracy by category
        category_accuracy = results["accuracy"]["by_category"]
        categories = list(category_accuracy.keys())
        accuracy_values = list(category_accuracy.values())
        
        plot_bar_chart(
            f"{arch_name} - Accuracy by Category",
            categories,
            accuracy_values,
            "Accuracy Score",
            os.path.join(viz_path, "accuracy_by_category.png"),
            threshold=self.config.accuracy_threshold,
            sort_values=True
        )
        
        # Performance metrics
        if EvaluationMetric.RESPONSE_LATENCY.value in results["metrics"]:
            latencies = results["metrics"][EvaluationMetric.RESPONSE_LATENCY.value]
            if isinstance(latencies, list) and latencies:
                scenarios = [f"Scenario {i+1}" for i in range(len(latencies))]
                plot_bar_chart(
                    f"{arch_name} - Response Latency by Scenario",
                    scenarios,
                    latencies,
                    "Latency (ms)",
                    os.path.join(viz_path, "response_latency.png")
                )
        
        # Memory usage
        if EvaluationMetric.MEMORY_USAGE.value in results["metrics"]:
            memory_usage = results["metrics"][EvaluationMetric.MEMORY_USAGE.value]
            if isinstance(memory_usage, list) and memory_usage:
                scenarios = [f"Scenario {i+1}" for i in range(len(memory_usage))]
                plot_bar_chart(
                    f"{arch_name} - Memory Usage by Scenario",
                    scenarios,
                    memory_usage,
                    "Memory Usage (MB)",
                    os.path.join(viz_path, "memory_usage.png")
                )
        
        # Accuracy by agent type
        if "by_agent_type" in results["accuracy"] and results["accuracy"]["by_agent_type"]:
            agent_accuracy = results["accuracy"]["by_agent_type"]
            agent_types = list(agent_accuracy.keys())
            accuracy_values = list(agent_accuracy.values())
            
            plot_bar_chart(
                f"{arch_name} - Accuracy by Agent Type",
                agent_types,
                accuracy_values,
                "Accuracy Score",
                os.path.join(viz_path, "accuracy_by_agent_type.png"),
                threshold=self.config.accuracy_threshold,
                sort_values=True
            )
        
        # Timeline visualization if we have execution timeline data
        if "execution_timeline" in results and results["execution_timeline"]:
            plot_timeline_visualization(
                results["execution_timeline"],
                os.path.join(viz_path, "execution_timeline.png"),
                title=f"{arch_name} - Agent Execution Timeline"
            )

    def _generate_comparative_visualizations(self):
        """Generate visualizations comparing all architectures"""
        if len(self.results) <= 1:
            return  # Need at least two architectures to compare
            
        viz_path = os.path.join(self.config.results_path, "comparative")
        os.makedirs(viz_path, exist_ok=True)
        
        # 1. Compare overall accuracy
        overall_accuracy = {
            arch: results["accuracy"]["overall_accuracy"]
            for arch, results in self.results.items()
        }
        
        plot_bar_chart(
            "Overall Accuracy by Architecture",
            list(overall_accuracy.keys()),
            list(overall_accuracy.values()),
            "Accuracy Score",
            os.path.join(viz_path, "overall_accuracy.png"),
            threshold=self.config.accuracy_threshold,
            color='lightgreen',
        )
        
        # 2. Compare execution time
        execution_time = {
            arch: results["metrics"][EvaluationMetric.EXECUTION_TIME.value]
            for arch, results in self.results.items()
            if EvaluationMetric.EXECUTION_TIME.value in results["metrics"]
        }
        
        plot_bar_chart(
            "Execution Time by Architecture",
            list(execution_time.keys()),
            list(execution_time.values()),
            "Execution Time (ms)",
            os.path.join(viz_path, "execution_time.png"),
            color='coral',
        )
        
        # 3. Compare category accuracy across architectures
        category_accuracy_by_arch = {}
        for arch, results in self.results.items():
            category_accuracy_by_arch[arch] = results["accuracy"]["by_category"]
        
        plot_multi_architecture_comparison(
            category_accuracy_by_arch,
            "Accuracy by Category",
            os.path.join(viz_path, "accuracy_by_category_comparison.png"),
            title="Accuracy by Category Across Architectures",
            ylabel="Accuracy Score",
            threshold=self.config.accuracy_threshold,
            sort_by="multi_framework" if "multi_framework" in category_accuracy_by_arch else None
        )
        
        # 4. Radar chart comparing key metrics
        if len(self.results) >= 2:
            # Select key metrics for radar chart
            radar_metrics = {
                arch: {
                    "Accuracy": results["accuracy"]["overall_accuracy"],
                    "Speed": 1.0 / (results["metrics"].get(EvaluationMetric.EXECUTION_TIME.value, 1) + 1),  # Invert so higher is better
                    "Memory": 1.0 / (np.mean(results["metrics"].get(EvaluationMetric.MEMORY_USAGE.value, [1])) + 1),  # Invert so higher is better
                    "Cold Start": 1.0 / (results["metrics"].get(EvaluationMetric.COLD_START_TIME.value, 1) + 1)  # Invert so higher is better
                }
                for arch, results in self.results.items()
            }
            
            # Add architecture-specific metrics if available
            for arch, results in self.results.items():
                if "accuracy_preservation" in results and "overall" in results["accuracy_preservation"]:
                    radar_metrics[arch]["Preservation"] = results["accuracy_preservation"]["overall"] / 100.0
                    
                if self.config.inter_agent_communication_tracking:
                    inter_agent_latency = results["metrics"].get(EvaluationMetric.INTER_AGENT_LATENCY.value)
                    if inter_agent_latency:
                        # Normalize and invert (lower latency is better)
                        max_latency = max(r["metrics"].get(EvaluationMetric.INTER_AGENT_LATENCY.value, 0) 
                                         for r in self.results.values() if EvaluationMetric.INTER_AGENT_LATENCY.value in r["metrics"])
                        if max_latency > 0:
                            radar_metrics[arch]["Communication"] = 1.0 - (inter_agent_latency / max_latency)
            
            metric_categories = list(set().union(*(metrics.keys() for metrics in radar_metrics.values())))
            
            plot_radar_chart(
                radar_metrics,
                metric_categories,
                os.path.join(viz_path, "architecture_radar_comparison.png"),
                title="Architecture Comparison - Key Metrics"
            )
        
        # 5. Plot accuracy preservation heatmap for hybrid architectures
        if "multi_framework" in self.results:
            baseline_accuracies = self.results["multi_framework"]["accuracy"]["by_category"]
            architecture_accuracies = {
                arch: results["accuracy"]["by_category"]
                for arch, results in self.results.items()
                if arch != "multi_framework"
            }
            
            if architecture_accuracies:  # If we have non-baseline architectures
                plot_accuracy_preservation_heatmap(
                    baseline_accuracies,
                    architecture_accuracies,
                    os.path.join(viz_path, "accuracy_preservation_heatmap.png"),
                    title="Accuracy Preservation Compared to Multi-Framework Baseline"
                )
        
        # 6. Plot performance vs accuracy trade-off
        performance_metrics = {
            arch: {
                "execution_time": results["metrics"].get(EvaluationMetric.EXECUTION_TIME.value, 0),
                "accuracy": results["accuracy"]["overall_accuracy"],
                "memory_usage": np.mean(results["metrics"].get(EvaluationMetric.MEMORY_USAGE.value, [0])),
                "cold_start_time": results["metrics"].get(EvaluationMetric.COLD_START_TIME.value, 0)
            }
            for arch, results in self.results.items()
        }
        
        plot_performance_vs_accuracy(
            performance_metrics,
            "execution_time",
            "accuracy",
            os.path.join(viz_path, "execution_time_vs_accuracy.png"),
            title="Execution Time vs Accuracy Trade-off",
            performance_label="Execution Speed (faster →)",
            accuracy_label="Accuracy Score",
            invert_performance=True
        )
        
        plot_performance_vs_accuracy(
            performance_metrics,
            "memory_usage",
            "accuracy",
            os.path.join(viz_path, "memory_usage_vs_accuracy.png"),
            title="Memory Usage vs Accuracy Trade-off",
            performance_label="Memory Efficiency (better →)",
            accuracy_label="Accuracy Score",
            invert_performance=True
        )

    def evaluate_specific_architecture(self, arch_name: str) -> Dict[str, Any]:
        """Evaluate a specific architecture by name"""
        if arch_name not in self.architectures:
            raise ValueError(f"Architecture {arch_name} not loaded")
            
        print(f"Evaluating {arch_name} architecture...")
        results = self._evaluate_architecture(arch_name)
        self.results[arch_name] = results
        
        return results

    def run_comparison(self, scenario_id: str, architecture_names: List[str]) -> Dict[str, Any]:
        """Run a specific scenario across multiple architectures for direct comparison"""
        if scenario_id not in self.ground_truth:
            raise ValueError(f"Scenario {scenario_id} not found in ground truth data")
            
        scenario = self.ground_truth[scenario_id]
        input_data = scenario["input"]
        expected_output = scenario["expected_output"]
        
        comparison_results = {}
        
        for arch_name in architecture_names:
            if arch_name not in self.architectures:
                print(f"Warning: Architecture {arch_name} not loaded, skipping")
                continue
                
            architecture = self.architectures[arch_name]
            
            # Initialize architecture
            architecture.initialize()
            
            # Execute with event tracking
            start_time = time.time()
            actual_output, events = architecture.run_with_events(
                input_data,
                track_communication=self.config.inter_agent_communication_tracking,
                track_serialization=self.config.state_serialization_tracking,
                track_adapters=self.config.adapter_overhead_tracking
            )
            end_time = time.time()
            
            # Calculate accuracy
            accuracy = calculate_accuracy_score(actual_output, expected_output)
            
            # Extract events for metrics
            communication_events = events.get("communication", [])
            serialization_events = events.get("serialization", [])
            adapter_events = events.get("adapters", [])
            execution_timeline = events.get("timeline", [])
            
            # Calculate key metrics
            metrics = {
                "execution_time_ms": (end_time - start_time) * 1000,
                "accuracy": accuracy
            }
            
            # Add communication metrics if available
            if communication_events:
                avg_latency, total_overhead, msg_count = measure_inter_agent_communication(communication_events)
                metrics["inter_agent_latency_ms"] = avg_latency
                metrics["message_overhead_bytes"] = total_overhead
                metrics["message_count"] = msg_count
            
            # Add serialization metrics if available
            if serialization_events:
                avg_time, avg_overhead = analyze_state_serialization(serialization_events)
                metrics["serialization_time_ms"] = avg_time
                metrics["serialization_overhead_bytes"] = avg_overhead
            
            # Add adapter metrics if available
            if adapter_events:
                avg_conversion, avg_execution, avg_switching = measure_adapter_overhead(adapter_events)
                metrics["adapter_conversion_time_ms"] = avg_conversion
                metrics["adapter_execution_time_ms"] = avg_execution
                metrics["framework_switching_latency_ms"] = avg_switching
            
            comparison_results[arch_name] = {
                "metrics": metrics,
                "output": actual_output,
                "timeline": execution_timeline
            }
        
        return comparison_results
