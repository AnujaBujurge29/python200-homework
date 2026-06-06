# Project 08 -- Cloud Cost Analysis & Video

## Cost Analysis Write-Up

**Scenario A (Lightweight compute):**

- Hourly rate (Standard_B1s): \$0.018
- Usage: 160 hours / month
- Monthly cost (approx): \$2.88

**Scenario B (Heavy analytics):**

- VM hourly rate (Standard_NC6s_v3): \$3.060
- VM usage: 730 hours / month
- Additional services: Azure SQL (General Purpose, 4 vCores), 1 TB Blob Storage
- Monthly cost (approx): \$2233.80

**Reflection:**

The lightweight Scenario A costs approximately the amount shown above (\$2.88). Scenario B is substantially more expensive — the GPU VM is the primary cost driver and can be an order of magnitude (or more) above the tiny B1s instance when run 24/7. If your numbers surprised you, note whether the VM, SQL, or storage contributed most to the total.

Interesting findings from the Pricing Calculator:

- You can quickly add multiple services and see how small changes (region, reserved instance vs pay-as-you-go, or number of vCores) impact monthly totals.
- Some managed services (e.g., managed databases, analytics services) can look cheaper at first but add fixed costs for storage or I/O that increase the final total.

**Script output and comparison:**

## Video Link

**[Video Link](https://youtu.be/Hizu51cxfsQ)**
