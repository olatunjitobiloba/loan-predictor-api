"""
Performance and load testing
"""

import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BASE_URL = "https://loan-predictor-api-91xu.onrender.com"  # Production URL


def test_endpoint(endpoint, method="GET", data=None):
    """Test single endpoint and measure response time"""
    start = time.time()

    if method == "GET":
        response = requests.get(f"{BASE_URL}{endpoint}")
    else:
        response = requests.post(f"{BASE_URL}{endpoint}", json=data)

    duration = time.time() - start

    return {
        "endpoint": endpoint,
        "status": response.status_code,
        "duration": duration,
        "success": response.status_code == 200,
    }


def load_test(endpoint, num_requests=100, concurrent=10, method="GET", data=None):
    """Run load test on endpoint"""
    print(f"\n{'='*70}")
    print(f"LOAD TEST: {method} {endpoint}")
    print(f"Requests: {num_requests}, Concurrent: {concurrent}")
    print(f"{'='*70}")

    results = []

    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = [
            executor.submit(test_endpoint, endpoint, method, data)
            for _ in range(num_requests)
        ]

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error: {e}")

    # Calculate statistics
    durations = [r["duration"] for r in results]
    successes = sum(1 for r in results if r["success"])

    print("\nResults:")
    print(f"  Total requests: {len(results)}")
    print(f"  Successful: {successes} ({successes/len(results)*100:.1f}%)")
    print(f"  Failed: {len(results) - successes}")
    print("\nResponse Times:")
    print(f"  Average: {statistics.mean(durations):.3f}s")
    print(f"  Median: {statistics.median(durations):.3f}s")
    print(f"  Min: {min(durations):.3f}s")
    print(f"  Max: {max(durations):.3f}s")
    print(f"  Std Dev: {statistics.stdev(durations):.3f}s")

    if len(durations) >= 20:
        print(f"  P95: {statistics.quantiles(durations, n=20)[18]:.3f}s")
        print(f"  P99: {statistics.quantiles(durations, n=100)[98]:.3f}s")

    print(f"\nThroughput: {len(results) / sum(durations):.2f} requests/second")

    return results


def test_caching_effectiveness():
    """Test if caching is working"""
    print(f"\n{'='*70}")
    print("CACHING EFFECTIVENESS TEST")
    print(f"{'='*70}")

    endpoint = "/statistics"

    # First request (cold cache)
    print("\n1. First request (cold cache)...")
    start = time.time()
    _ = requests.get(f"{BASE_URL}{endpoint}")
    duration1 = time.time() - start
    print(f"   Duration: {duration1:.3f}s")

    # Second request (should be cached)
    print("\n2. Second request (should be cached)...")
    start = time.time()
    _ = requests.get(f"{BASE_URL}{endpoint}")
    duration2 = time.time() - start
    print(f"   Duration: {duration2:.3f}s")

    # Calculate improvement
    if duration1 > 0:
        improvement = ((duration1 - duration2) / duration1) * 100
        print(f"\n✅ Cache speedup: {improvement:.1f}% faster")
        print(f"   Speedup factor: {duration1/duration2:.1f}x")

    return duration1, duration2


def main():
    """Run all performance tests"""
    print("=" * 70)
    print("PERFORMANCE TEST SUITE")
    print("=" * 70)

    # Test 1: Caching effectiveness
    test_caching_effectiveness()

    # Test 2: Load test GET endpoints
    load_test("/health", num_requests=50, concurrent=10)
    load_test("/statistics", num_requests=50, concurrent=10)
    load_test("/model-info", num_requests=50, concurrent=10)

    # Test 3: Load test POST endpoint (predictions)
    prediction_data = {"ApplicantIncome": 5000, "LoanAmount": 150, "Credit_History": 1}
    load_test(
        "/predict", num_requests=20, concurrent=5, method="POST", data=prediction_data
    )

    # Test 4: Check performance metrics endpoint
    print(f"\n{'='*70}")
    print("PERFORMANCE METRICS")
    print(f"{'='*70}")
    response = requests.get(f"{BASE_URL}/performance")
    if response.status_code == 200:
        metrics = response.json()
        print("\nAPI Performance Metrics:")
        print(f"  Requests tracked: {metrics.get('requests_tracked')}")
        print(f"  Avg response time: {metrics.get('avg_response_time')}")
        print(f"  P95 response time: {metrics.get('p95_response_time')}")
        print(f"  P99 response time: {metrics.get('p99_response_time')}")
        print(f"  Cache: {metrics.get('cache_info', {}).get('type')}")
        print(f"  Compression: {metrics.get('compression')}")

    print(f"\n{'='*70}")
    print("✅ PERFORMANCE TESTS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
