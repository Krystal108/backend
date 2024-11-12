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
-- Database: `hr_management`
--

-- --------------------------------------------------------

--
-- Table structure for table `worker_evaluations`
--

CREATE TABLE `worker_evaluations` (
  `id` text NOT NULL,
  `employee_id` text NOT NULL,
  `name` text NOT NULL,
  `position` text NOT NULL,
  `department` text NOT NULL,
  `start_date` date NOT NULL,
  `comments` text NOT NULL,
  `performance` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `worker_evaluations`
--

INSERT INTO `worker_evaluations` (`id`, `employee_id`, `name`, `position`, `department`, `start_date`, `comments`, `performance`) VALUES
('EV5009', 'test', 'test', 'test', 'Warehouse', '2024-09-30', 'testest', 50),
('EV4049', 'bb', 'Azrael', 'sffs', 'Purchasing', '2024-09-24', '', 33),
('EV2508', 'PV-23234', 'Emily', 'Manager', 'Logistics', '2024-09-29', 'sdfsdfsdfsd', 10);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
