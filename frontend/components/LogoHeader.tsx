"use client";

export default function LogoHeader() {
  return (
    <div className="mt-6 text-center">
      {/* Centered Large Logo */}
      <div className="inline-flex items-center justify-center">
        <img
          src="/assets/aion_full_logo.png"
          alt="AION Analytics"
          className="h-40 md:h-48 w-auto mx-auto"
        />
      </div>
    </div>
  );
}
