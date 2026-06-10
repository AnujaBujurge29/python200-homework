# Week 8 - Part 1: Warmup -- Check for Understanding

## Cloud Concepts:

### Cloud Concepts Question 1

#### **Qu: What is the core economic model of cloud computing, and how does it differ from owning your own servers?**

**Answer:**

Cloud computing follows a pay-as-you-go (utility) model where you rent computing resources on demand and only pay for what you use, similar to how you pay for electricity. This differs from owning servers (capital expenditure model) where you must buy, house, and maintain physical hardware upfront -- paying the full cost whether you use it all or not. The cloud shifts spending from large upfront capital expenses to smaller, ongoing operational expenses that scale with actual usage.

### Cloud Concepts Question 2

#### **Qu: What is the difference between vertical scaling and horizontal scaling? Give a concrete example of when you might choose each.**

**Answer:**

Vertical scaling (scaling up) means making a single machine more powerful by adding more CPU, RAM, or a faster GPU. You choose vertical scaling when the workload cannot be easily split across multiple machines -- for example, a single database server that needs more memory to cache a growing dataset.

Horizontal scaling (scaling out) means adding more machines to share the workload. You choose horizontal scaling when the work can be divided into independent pieces -- for example, adding more web servers behind a load balancer to handle growing user traffic.

**Scenarios:**

- **A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch.**

  This is horizontal scaling because you can add more web server instances behind a load balancer to distribute the increased user traffic across multiple machines.

* **A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM.**

  This is vertical scaling because the solution is upgrading a single machine to have more powerful hardware (faster GPU, more RAM) rather than distributing the job.

- **A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machines.**

  This is horizontal scaling because the file processing work is independent and can be divided across multiple machines running in parallel.

### Cloud Concepts Question 3

**Classifications:**

- **Gmail** -- SaaS

  It is a fully managed application delivered through the browser; users just log in and use it with no infrastructure or platform management needed.

- **Azure Virtual Machines** -- IaaS.

  You get a raw virtual machine and are responsible for the OS, software, and everything above the hardware layer.

- **Azure App Service** -- PaaS.

  Azure manages the underlying servers and OS; you just deploy your application code.

- **AWS S3 (Simple Storage Service)** -- IaaS.

  It provides raw storage infrastructure that you manage and organize yourself (buckets, permissions, lifecycle rules).

- **GitHub Codespaces** -- PaaS.

  It provides a managed development environment where the underlying compute and OS are handled for you; you focus on your code.

- **Snowflake** -- SaaS (or managed PaaS/data platform).
  Users interact with it through a web interface and SQL queries without managing any underlying infrastructure.

**Definitions:**

- **IaaS (Infrastructure as a Service):**

  The cloud provider gives you virtual hardware -- servers, storage, networking -- and you manage everything from the operating system up. Example: Azure Virtual Machines. As the developer, you are responsible for the OS, security patches, runtime, middleware, and your application.

* **PaaS (Platform as a Service):**

  The provider manages the infrastructure and platform (OS, runtime, middleware), and you just deploy your code. Example: Azure App Service. As the developer, you are responsible for your application code and its configuration, but not the servers or OS underneath.

- **SaaS (Software as a Service):**

  The provider delivers a complete, ready-to-use application over the internet. Example: Gmail. As the developer/user, you are only responsible for your data and how you use the application -- everything else is managed for you.

### Cloud Concepts Question 4:

#### **What is a managed data platform like Databricks or Snowflake, and how does it differ from using a cloud provider like Azure directly? What do you gain, and what do you give up?**

**Answer:**

A managed data platform is a specialized service built on top of a cloud provider's infrastructure that handles the complexity of data engineering, analytics, or machine learning workflows for you. Unlike using Azure directly (where you would provision VMs, install Spark, manage clusters, etc.), these platforms abstract away infrastructure management and offer optimized tools for data work.

**What you gain:**

Ease of use, automatic scaling, built-in optimization, less operational overhead, and faster time to value for data workloads.

**What you give up:**

Fine-grained control over the underlying infrastructure, potential vendor lock-in, and typically higher per-unit costs compared to managing the same tools yourself on raw cloud resources.

### Cloud Concepts Question 5:

#### **The lesson names two situations where the cloud is probably not the right choice. What are they?**

**Answer:**

1. **When you have a steady, predictable workload that runs 24/7** --

   if your servers are always fully utilized, owning hardware may be cheaper in the long run than paying ongoing cloud rental costs.

2. **When you have strict data-residency or regulatory requirements** --

   some organizations must keep data on premises due to legal, compliance, or security regulations that cloud providers cannot satisfy.

## Azure Basics

### Azure Basics Question 1:

#### **What is the difference between an Azure subscription and a resource group? Which one is yours alone, and which one does CTD share?**

**Answer:**

An Azure subscription is the billing and access boundary -- it is the top-level container that determines who pays for resources and who can manage them. A resource group is a logical container within a subscription used to organize related resources (VMs, databases, storage) together.

CTD shares one subscription (the bill goes to CTD's account), and each student gets their own resource group within that subscription to organize their individual resources.

### Azure Basics Question 2

#### **Azure Cloud Shell is ephemeral by default. What does that mean in practice, and what does your course setup use to make it persistent?**

**Answer:**

Ephemeral means that the Cloud Shell container is temporary -- when your session ends or times out, the underlying machine is recycled and any files or installed software outside of your home directory are lost. The course setup uses an Azure File Share mounted to your Cloud Shell home directory (`$HOME`) to make your files persistent across sessions, so your scripts, SSH keys, and configuration files survive even when the container is recreated.

### Azure Basics Question 3

#### **What is the difference between your SSH private key and your SSH public key? Which one gets uploaded to the remote systems you want to connect to, and why is that safe?**

**Answer:**

Your SSH private key is your secret -- it stays on your local machine and is never shared. Your SSH public key is the matching counterpart that can be freely distributed. The public key gets uploaded to the remote systems you want to connect to. This is safe because the public key can only verify your identity (confirm that you hold the matching private key); it cannot be used to impersonate you or derive the private key. It is like giving someone a lock -- they can verify you have the key, but they cannot make a copy of it.

### Azure Basics Question 4:

**Run `az account show` without `--output table`:**

Output JSON:

```json
{
  "environmentName": "AzureCloud",
  "homeTenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "id": "4e07c58c-751e-4765-b40c-632b9ee6fe6e",
  "isDefault": true,
  "managedByTenants": [],
  "name": "CTD Nonprofit Sponsorship",
  "state": "Enabled",
  "tenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "user": {
    "cloudShellID": true,
    "name": "live.com#anuja.bujurge29@gmail.com",
    "type": "user"
  }
}
```

When you add `--output table`, Azure CLI formats the same information as a simplified, human-readable table with columns and rows instead of the default verbose JSON output.

**Run `az account show --output table`:**

| EnvironmentName | HomeTenantId                         | IsDefault | Name                      | State   | TenantId                             |
| --------------- | ------------------------------------ | --------- | ------------------------- | ------- | ------------------------------------ |
| AzureCloud      | 0f040ddd-301f-4665-8677-7b21f129d605 | True      | CTD Nonprofit Sponsorship | Enabled | 0f040ddd-301f-4665-8677-7b21f129d605 |
