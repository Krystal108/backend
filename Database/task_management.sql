-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1:3307
-- Generation Time: Sep 28, 2024 at 03:18 PM
-- Server version: 10.4.32-MariaDB
-- PHP Version: 8.0.30

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `task_management`
--

-- --------------------------------------------------------

--
-- Table structure for table `accounting_tasks`
--

CREATE TABLE `accounting_tasks` (
  `id` int(11) NOT NULL,
  `task` text NOT NULL,
  `owner` text NOT NULL,
  `status` text NOT NULL,
  `start_date` date NOT NULL,
  `due_date` date NOT NULL,
  `completion` int(11) NOT NULL,
  `priority` text NOT NULL,
  `duration` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `logistics_tasks`
--

CREATE TABLE `logistics_tasks` (
  `id` int(11) NOT NULL,
  `task` text NOT NULL,
  `owner` text NOT NULL,
  `status` text NOT NULL,
  `start_date` date NOT NULL,
  `due_date` date NOT NULL,
  `completion` int(11) NOT NULL,
  `priority` text NOT NULL,
  `duration` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `proddev_tasks`
--

CREATE TABLE `proddev_tasks` (
  `id` int(11) NOT NULL,
  `task` text NOT NULL,
  `owner` text NOT NULL,
  `status` text NOT NULL,
  `start_date` date NOT NULL,
  `due_date` date NOT NULL,
  `completion` int(11) NOT NULL,
  `priority` text NOT NULL,
  `duration` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `purchasing_tasks`
--

CREATE TABLE `purchasing_tasks` (
  `id` int(11) NOT NULL,
  `task` text NOT NULL,
  `owner` text NOT NULL,
  `status` text NOT NULL,
  `start_date` date NOT NULL,
  `due_date` date NOT NULL,
  `completion` int(11) NOT NULL,
  `priority` text NOT NULL,
  `duration` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `purchasing_tasks`
--

INSERT INTO `purchasing_tasks` (`id`, `task`, `owner`, `status`, `start_date`, `due_date`, `completion`, `priority`, `duration`) VALUES
(1, 'test', 'test', 'Not Started', '2024-09-18', '2024-09-25', 5, 'Medium', 5);

-- --------------------------------------------------------

--
-- Table structure for table `sales_tasks`
--

CREATE TABLE `sales_tasks` (
  `id` text NOT NULL,
  `task` text NOT NULL,
  `owner` text NOT NULL,
  `status` text NOT NULL,
  `start_date` date NOT NULL,
  `due_date` date NOT NULL,
  `completion` int(11) NOT NULL,
  `priority` text NOT NULL,
  `duration` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `sales_tasks`
--

INSERT INTO `sales_tasks` (`id`, `task`, `owner`, `status`, `start_date`, `due_date`, `completion`, `priority`, `duration`) VALUES
('PC-T8797', 'a', 'ff', 'Not Started', '2024-09-24', '2024-09-27', 35, 'High', 8),
('PC-T7953', 'b', 'b', 'In Progress', '2024-09-22', '2024-09-28', 80, 'Medium', 7);

-- --------------------------------------------------------

--
-- Table structure for table `warehouse_tasks`
--

CREATE TABLE `warehouse_tasks` (
  `id` int(11) NOT NULL,
  `task` text NOT NULL,
  `owner` text NOT NULL,
  `status` text NOT NULL,
  `start_date` date NOT NULL,
  `due_date` date NOT NULL,
  `completion` int(11) NOT NULL,
  `priority` text NOT NULL,
  `duration` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Indexes for dumped tables
--

--
-- Indexes for table `accounting_tasks`
--
ALTER TABLE `accounting_tasks`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `id` (`id`);

--
-- Indexes for table `logistics_tasks`
--
ALTER TABLE `logistics_tasks`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `id` (`id`);

--
-- Indexes for table `proddev_tasks`
--
ALTER TABLE `proddev_tasks`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `id` (`id`);

--
-- Indexes for table `purchasing_tasks`
--
ALTER TABLE `purchasing_tasks`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `id` (`id`);

--
-- Indexes for table `warehouse_tasks`
--
ALTER TABLE `warehouse_tasks`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `id` (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `accounting_tasks`
--
ALTER TABLE `accounting_tasks`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `logistics_tasks`
--
ALTER TABLE `logistics_tasks`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `proddev_tasks`
--
ALTER TABLE `proddev_tasks`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `purchasing_tasks`
--
ALTER TABLE `purchasing_tasks`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT for table `warehouse_tasks`
--
ALTER TABLE `warehouse_tasks`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
